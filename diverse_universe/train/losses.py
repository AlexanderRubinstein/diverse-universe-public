import torch
import sys
from torch.distributions.categorical import Categorical
from stuned.utility.utils import (
    get_project_root_path,
    aggregate_tensors_by_func,
    func_for_dim,
    apply_pairwise,
    get_with_assert,
    raise_unknown
)
from stuned.utility.logger import (
    make_logger
)
import torch.nn.functional as F
from einops import rearrange


# local modules
sys.path.insert(0, get_project_root_path())
from diverse_universe.train.utils import (
    take_from_2d_tensor,
)
from diverse_universe.local_models.utils import (
    compute_ensemble_output,
    bootstrap_ensemble_outputs
)
from diverse_universe.local_models.wrappers import (
    make_mvh_mapper
)
sys.path.pop(0)


LOSS_STATISTICS_NAME = "loss"
DIVDIS_LOSS_KEY = "divdis"
LAMBDA_KEY = "lambda"
EPS = 1e-9
MAX_MODELS_WITHOUT_OOM = 5


PER_SAMPLE_METRIC_NAMES = [
    "div_different_preds_per_sample",
    "div_continous_unique_per_sample",
    "ens_entropy_per_sample",
    "average_entropy_per_sample",
    "mutual_information_per_sample",
    "average_energy_per_sample",
    "average_max_logit_per_sample",
    "a2d_score_per_sample",
    "similarity_between_models_per_sample",
    "unreduced_entropy_per_sample"
]


STACKED_INPUTS_METRIC_NAMES = [
    "var",
    "std",
    "dis",
    "max_var",
    "div_different_preds",
    "div_mean_logits",
    "div_max_logit",
    "div_entropy",
    "div_max_prob",
    "div_mean_prob",
    "div_different_preds_per_model",
    "div_continous_unique",
    "ens_entropy",
    "average_entropy",
    "mutual_information",
    "average_energy",
    "average_max_logit",
    "a2d_score"
]


# copied from: https://github.com/yoonholee/DivDis/blob/main/divdis.py#L6
def to_probs(logits, heads):
    """
    Converts logits to probabilities.
    Input must have shape [batch_size, heads * classes].
    Output will have shape [batch_size, heads, classes].
    """

    B, N = logits.shape
    if N == heads:  # Binary classification; each head outputs a single scalar.
        preds = logits.sigmoid().unsqueeze(-1)
        probs = torch.cat([preds, 1 - preds], dim=-1)
    else:
        logits_chunked = torch.chunk(logits, heads, dim=-1)
        probs = torch.stack(logits_chunked, dim=1).softmax(-1)
    B, H, D = probs.shape
    assert H == heads
    return probs


# copied from: https://github.com/yoonholee/DivDis/blob/main/divdis.py#L46
class DivDisLoss(torch.nn.Module):
    """Computes pairwise repulsion losses for DivDis.

    Args:
        logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS * DIM].
        heads (int): Number of heads.
        mode (str): DIVE loss mode. One of {pair_mi, total_correlation, pair_l1}.
    """

    def __init__(self, heads, mode="mi", reduction="mean"):
        super().__init__()
        self.heads = heads
        self.mode = mode
        self.reduction = reduction

    def forward(self, logits):
        heads, mode, reduction = self.heads, self.mode, self.reduction
        probs = to_probs(logits, heads)
        return divdis_loss_forward_impl(probs, mode=mode, reduction=reduction)


# based on: https://github.com/yoonholee/DivDis/blob/main/divdis.py
def divdis_loss_forward_impl(probs, mode="mi", reduction="mean"):
    # input has shape [batch_size, heads, classes]
    heads = probs.shape[1]

    if mode == "mi":  # This was used in the paper
        marginal_p = probs.mean(dim=0)  # H, D
        marginal_p = torch.einsum(
            "hd,ge->hgde", marginal_p, marginal_p
        )  # H, H, D, D
        marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # H^2, D^2

        joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(
            dim=0
        )  # H, H, D, D
        joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # H^2, D^2

        # Compute pairwise mutual information = KL(P_XY | P_X x P_Y)
        # Equivalent to: F.kl_div(marginal_p.log(), joint_p, reduction="none")
        kl_computed = joint_p * (joint_p.log() - marginal_p.log() + EPS)
        kl_computed = kl_computed.sum(dim=-1)
        kl_grid = rearrange(kl_computed, "(h g) -> h g", h=heads)
        repulsion_grid = -kl_grid
    elif mode == "l1":
        dists = (probs.unsqueeze(1) - probs.unsqueeze(2)).abs()
        dists = dists.sum(dim=-1).mean(dim=0)
        repulsion_grid = dists
    else:
        raise ValueError(f"{mode=} not implemented!")

    if reduction == "mean":  # This was used in the paper
        repulsion_grid = torch.triu(repulsion_grid, diagonal=1)
        repulsions = repulsion_grid[repulsion_grid.nonzero(as_tuple=True)]
        repulsion_loss = -repulsions.mean()
    elif reduction == "min_each":
        repulsion_grid = torch.triu(repulsion_grid, diagonal=1) + torch.tril(
            repulsion_grid, diagonal=-1
        )
        rows = [r for r in repulsion_grid]
        row_mins = [row[row.nonzero(as_tuple=True)].min() for row in rows]
        repulsion_loss = -torch.stack(row_mins).mean()
    else:
        raise ValueError(f"{reduction=} not implemented!")

    return repulsion_loss


def budget_modifier(outputs, target):
    with torch.no_grad():
        ensemble_output = compute_ensemble_output(outputs)

        unreduced_ce = F.cross_entropy(
            ensemble_output,
            target,
            reduction='none'
        )
        divider = torch.pow(unreduced_ce.mean(0), 2)
        return unreduced_ce / divider


class DivDisLossWrapper(torch.nn.Module):

    def __init__(
        self,
        task_loss,
        weight,
        mode="mi",
        reduction="mean",
        mapper=None,
        loss_type="divdis",
        use_always_labeled=False,
        modifier=None,
        gamma=2.0,
        disagree_after_epoch=0,
        manual_lambda=1.0,
        disagree_below_threshold=None,
        reg_mode=None,
        reg_weight=None
    ):
        super().__init__()
        self.repulsion_loss = None
        self.mode = mode
        self.reduction = reduction
        self.task_loss = task_loss
        self.weight = weight
        self.loss_type = loss_type
        if mapper == "mvh":
            self.mapper = make_mvh_mapper()
        else:
            assert mapper is None, "Only mvh mapper is supported"
            self.mapper = None
        self.log_this_batch = True
        self.use_always_labeled = use_always_labeled
        self.modifier = modifier
        self.gamma = gamma
        self.epoch = 0
        self.disagree_after_epoch = disagree_after_epoch
        self.manual_lambda = manual_lambda
        self.disagree_below_threshold = disagree_below_threshold
        self.reg_mode = reg_mode
        self.reg_weight = reg_weight

        if self.reg_mode is not None:
            assert self.reg_weight is not None

    def increase_epoch(self):
        self.epoch += 1

    def compute_modifier(self, outputs, targets):

        if self.modifier == "budget":
            return budget_modifier(outputs, targets)
        else:
            assert self.modifier is None
            return 1.0

    def forward(self, outputs, targets, unlabeled_outputs=None):

        def zero_if_none(value):
            return (
                value.item()
                    if value is not None
                    else 0
            )

        def get_repulsion_loss(outputs, unlabeled_outputs, targets_values):
            if unlabeled_outputs is None:
                assert not outputs[0][1].requires_grad, \
                    "No unlabeled batch was provided during training"
                repulsion_loss = None
                modifier = None
            else:
                n_heads = len(unlabeled_outputs)
                if self.repulsion_loss is None:

                    if self.loss_type == "divdis":
                        assert self.modifier != "budget", \
                            "Budget modifier is not supported for DivDisLoss"
                        self.repulsion_loss = DivDisLoss(
                            n_heads,
                            self.mode,
                            self.reduction
                        )
                    else:
                        assert self.loss_type == "a2d"
                        reduction = "mean"
                        if self.modifier == "budget":
                            reduction = "none"
                        self.repulsion_loss = A2DLoss(
                            n_heads,
                            reduction=reduction
                        )
                else:
                    self.repulsion_loss.heads = n_heads

                if self.use_always_labeled:
                    modifier = self.compute_modifier(
                        unlabeled_outputs,
                        targets_values
                    )
                else:
                    assert self.modifier is None
                    modifier = 1.0

                if self.mapper is not None:
                    unlabeled_outputs \
                        = [self.mapper(output) for output in unlabeled_outputs]

                # [batch, n * classes]
                unlabeled_outputs_cat = torch.cat(
                    unlabeled_outputs,
                    axis=-1
                )

                repulsion_loss = self.repulsion_loss(unlabeled_outputs_cat)
            return repulsion_loss, modifier, unlabeled_outputs

        if self.use_always_labeled:
            assert unlabeled_outputs is None
            unlabeled_outputs = outputs

        cur_n_heads = len(outputs)

        metrics_mappings = get_metrics_mapping(
            cur_n_heads > MAX_MODELS_WITHOUT_OOM
        )

        outputs = bootstrap_ensemble_outputs(outputs)
        targets_values = targets.max(-1).indices
        for output in outputs:
            assert not torch.isnan(output).any(), "NaNs in outputs"
        if unlabeled_outputs is not None:
            unlabeled_outputs = bootstrap_ensemble_outputs(unlabeled_outputs)

        repulsion_loss, modifier, reg_loss = None, None, None

        if self.weight > 0 and self.epoch + 1 > self.disagree_after_epoch:
            if self.disagree_below_threshold is not None:
                assert self.modifier is None, \
                    "Can't use modifier with disagree_below_threshold"
                assert self.weight < 1, \
                    "Can't have lambda == 1 with disagree_below_threshold"
                assert self.use_always_labeled

                masks = [
                    (
                            take_from_2d_tensor(
                                get_probs(output),
                                targets_values,
                                dim=-1
                            )
                        >
                            self.disagree_below_threshold
                    )
                        for output
                        in outputs
                ]
                # take samples which are low prob for all models
                mask = torch.stack(masks).min(0).values
                unlabeled_outputs = [
                    output[~mask, ...]
                        for output
                        in outputs
                ]
                outputs = [output[mask, ...] for output in outputs]
                targets = targets[mask, ...]
            if (
                    unlabeled_outputs is not None
                and
                    len(unlabeled_outputs) > 0
                and
                    len(unlabeled_outputs[0]) > 0
            ):
                repulsion_loss, modifier, unlabeled_outputs = get_repulsion_loss(
                    outputs,
                    unlabeled_outputs,
                    targets_values
                )

                reg_loss = get_regularizer(
                    self.reg_mode,
                    outputs,
                    unlabeled_outputs
                )

        if repulsion_loss is not None:
            assert not torch.isnan(repulsion_loss).any(), "NaNs in repulsion_loss"

        task_loss_value = torch.Tensor([0])[0]
        total_loss = task_loss_value.to(targets.device)
        if self.weight < 1:
            if len(outputs) > 0 and len(outputs[0]) > 0:
                task_loss_value = aggregate_tensors_by_func(
                    [self.task_loss(output, targets) for output in outputs]
                )

                total_loss = (1 - self.weight) * task_loss_value
        else:
            assert self.weight == 1
            assert self.disagree_after_epoch == 0, \
                "When lambda is 1 disagreement should start from the first epoch"

        if repulsion_loss is not None:

            repulsion_loss *= modifier

            if len(repulsion_loss.shape) > 0 and repulsion_loss.shape[0] > 1:
                repulsion_loss = repulsion_loss.mean()

            total_loss += self.manual_lambda * self.weight * repulsion_loss

        if reg_loss is not None:
            total_loss += self.reg_weight * self.weight * reg_loss

        loss_info = {
            "task_loss": task_loss_value.item(),
            "repulsion_loss": zero_if_none(repulsion_loss),
            "regularizer_loss": zero_if_none(reg_loss),
            "total_loss": total_loss.item()
        }

        if self.log_this_batch:
            record_diversity(
                loss_info,
                outputs,
                torch.stack(outputs),
                metrics_mappings=metrics_mappings,
                name_prefix="ID_loss_"
            )
            if unlabeled_outputs is not None and not self.use_always_labeled:

                record_diversity(
                    loss_info,
                    unlabeled_outputs,
                    torch.stack(unlabeled_outputs),
                    metrics_mappings=metrics_mappings,
                    name_prefix="OOD_loss_"
                )
        gradients_info = {}
        return total_loss, loss_info, gradients_info


# inspired by this: https://github.com/yoonholee/DivDis/blob/b9de1a637949594054240254f667063788ee1573/subpopulation/train.py#L197-L220
def get_regularizer(reg_mode, outputs, unlabeled_outputs):

    def chunk(outputs, heads):
        outputs_cat = torch.cat(
            outputs,
            axis=-1
        )
        chunked = torch.chunk(outputs_cat, heads, dim=-1)
        return chunked

    if reg_mode is None:
        return None

    assert reg_mode == "kl_backward"
    heads = len(outputs)

    yhat_chunked = chunk(outputs, heads)
    yhat_unlabeled_chunked = chunk(unlabeled_outputs, heads)

    preds = torch.stack(yhat_unlabeled_chunked).softmax(-1)

    avg_preds_source = (
        torch.stack(yhat_chunked).softmax(-1).mean([0, 1]).detach()
    )
    avg_preds_target = preds.mean(1)
    dist_source = Categorical(probs=avg_preds_source)
    dist_target = Categorical(probs=avg_preds_target)
    if reg_mode in ["kl_forward", "kl_ratio_f"]:
        kl = torch.distributions.kl.kl_divergence(dist_source, dist_target)
    elif reg_mode in ["kl_backward", "kl_ratio_b"]:
        kl = torch.distributions.kl.kl_divergence(dist_target, dist_source)
    reg_loss = kl.mean()
    return reg_loss


def make_divdis_loss(divdis_loss_config, logger):
    task_loss = make_criterion(
        get_with_assert(divdis_loss_config, "task_loss"),
        logger=logger
    )
    repulsion_loss_type = divdis_loss_config.get("loss_type", "divdis")
    logger.log(
        "Using repulsion_loss \"{}\" for DivDisLoss".format(repulsion_loss_type)
    )
    mapper = divdis_loss_config.get("mapper")
    reg_mode = divdis_loss_config.get("reg_mode")
    if reg_mode is not None:
        logger.log(
            f"Using regularizer {reg_mode}"
        )
    if mapper is not None:
        logger.log("Using mapper \"{}\" for DivDisLoss".format(mapper))
    return DivDisLossWrapper(
        task_loss,
        get_with_assert(divdis_loss_config, "lambda"),
        divdis_loss_config.get("mode", "mi"),
        divdis_loss_config.get("reduction", "mean"),
        mapper=mapper,
        loss_type=repulsion_loss_type,
        use_always_labeled=divdis_loss_config.get("use_always_labeled", False),
        modifier=divdis_loss_config.get("modifier"),
        gamma=divdis_loss_config.get("gamma", 2.0),
        disagree_after_epoch=divdis_loss_config.get("disagree_after_epoch", 0),
        manual_lambda=divdis_loss_config.get("manual_lambda", 1.0),
        disagree_below_threshold=divdis_loss_config.get(
            "disagree_below_threshold",
            None
        ),
        reg_mode=reg_mode,
        reg_weight=divdis_loss_config.get("reg_weight")
    )


def make_criterion(
    criterion_config,
    cache_path=None,
    logger=None,
    device="cpu"
):
    if logger is None:
        logger = make_logger()

    criterion_type = criterion_config["type"]
    logger.log("Making criterion \"{}\"..".format(criterion_type))
    if criterion_type == "bce":
        criterion = torch.nn.BCELoss()
    elif criterion_type == "xce":
        criterion = torch.nn.CrossEntropyLoss()
    elif criterion_type == DIVDIS_LOSS_KEY:
        criterion = make_divdis_loss(criterion_config[criterion_type], logger)
    else:
        raise_unknown("criterion", criterion_type, "criterion config")

    criterion.smoothing_eps = criterion_config.get("smoothing_eps")

    return criterion


# adapted from: https://github.com/Guoxoug/ens-div-ood-detect/blob/fd52391cb10023d9648b1bd7947530dc48e860a2/utils/eval_utils.py#L117

def entropy(probs: torch.Tensor, dim=-1):
    "Calcuate the entropy of a categorical probability distribution."
    log_probs = probs.log()
    ent = (-probs*log_probs).sum(dim=dim)
    return ent


def ens_entropy_per_sample(logits, models_dim=0):
    probs = get_probs(logits)
    av_probs = probs.mean(dim=models_dim)
    ent = entropy(av_probs)
    return ent


def ens_entropy(logits, models_dim=0):
    return ens_entropy_per_sample(logits, models_dim=models_dim).mean()


def pairwise_distances(stacked_vectors, p=2):
    distances = torch.cdist(stacked_vectors, stacked_vectors, p=p)

    # Step 2: Calculate the average pairwise distance
    # We only need the upper triangle of the matrix (excluding the diagonal)
    # to avoid duplicate distances and to exclude the distance of vectors
    # with themselves (which is zero)

    # Compute the mean of these distances
    average_distance = distances.mean((-1, -2))
    return average_distance


def similarity_between_models_per_sample(logits):

    probs = get_probs(logits)
    probs_batch_first = probs.transpose(0, 1)
    similarity = pairwise_distances(probs_batch_first)
    return similarity


def unreduced_entropy_per_sample(logits):
    return average_entropy_per_sample(logits, models_dim=None).transpose(0, 1)


def average_entropy_per_sample(logits, models_dim=0):
    probs = get_probs(logits)

    av_ent = entropy(probs, dim=-1)

    if models_dim is not None:
        av_ent = av_ent.mean(dim=models_dim)

    return av_ent


def average_entropy(logits, models_dim=0):
    return average_entropy_per_sample(logits, models_dim=models_dim).mean()


def mutual_information_per_sample(logits, models_dim=0):
    probs = get_probs(logits)
    av_probs = probs.mean(dim=models_dim)
    ent = entropy(av_probs)
    av_ent = average_entropy(logits)
    mutual_information = ent - av_ent
    return mutual_information


def a2d_score(logits):
    return a2d_score_per_sample(logits).mean()


# input shape: [n_models, batch_size, num_classes]
def a2d_score_per_sample(logits):
    probs = get_probs(logits)
    total_score = None

    # for symmetry treat each model as p2
    for m_idx in range(probs.shape[0]):
        a2d_loss = a2d_loss_impl(
            probs,
            m_idx,
            dbat_loss_type='v1',
            reduction='none'
        )
        inv_a2d_loss = -a2d_loss
        if total_score is None:
            total_score = inv_a2d_loss.unsqueeze(0)
        else:
            total_score = torch.cat([total_score, inv_a2d_loss.unsqueeze(0)], dim=0)

    return total_score.mean(0)


def mutual_information(logits):
    return mutual_information_per_sample(logits).mean()


def average_energy_per_sample(logits, models_dim=0):
    return -torch.logsumexp(logits, dim=-1).mean(dim=models_dim)


def average_energy(logits, models_dim=0):
    return average_energy_per_sample(logits, models_dim=models_dim).mean()


def average_max_logit_per_sample(logits, models_dim=0):
    return -(logits.max(dim=-1).values.mean(dim=models_dim))


def average_max_logit(logits, models_dim=0):
    return average_max_logit_per_sample(logits, models_dim=models_dim).mean()


def conf(logits, models_dim=0):
    probs = logits.softmax(dim=-1)
    av_probs = probs.mean(dim=models_dim)
    conf = av_probs.max(dim=-1).values
    return conf.mean()


def div_different_preds(logits):

    return div_different_preds_per_sample(logits).mean()


def div_continous_unique(logits):

    return div_continous_unique_per_sample(logits).mean()


# Called Predictive Diversity Score (PDS) in paper
def div_continous_unique_per_sample(logits):
    probs = get_probs(logits).clone()

    max_by_model = probs.max(0).values
    sum_over_classes = max_by_model.sum(-1)

    return torch.Tensor(sum_over_classes)


def div_different_preds_per_sample(logits):

    probs = get_probs(logits)
    preds = probs.argmax(-1).t()
    res = []

    for per_sample_preds in preds:
        res.append(len(per_sample_preds.unique()))

    return torch.Tensor(res)


def div_var(outputs):
    return -F.softmax(outputs, dim=-1).var(0).mean()


def max_prob_var(outputs):

    probs = F.softmax(outputs, dim=-1)
    max_across_models = torch.max(probs, dim=-1).indices
    probs = probs[..., max_across_models]

    return probs.var(0).mean()


def div_std(outputs):
    return -F.softmax(outputs, dim=-1).std(0).mean()


def kl_divergence(out1, out2):
    loss = F.kl_div(
        torch.log(F.softmax(out1, dim=-1) + 1e-8),
        F.softmax(out2, dim=-1) + 1e-8,
        reduction='batchmean'
    )
    return -loss  # negative because we want to maximize KL divergence


def dis(out, mode="mi", reduction="mean"):
    """
    Calculate the loss to minimize the mutual information between all pairs of models.
    `out` should have shape [MODELS, BATCH, NUM_CLASSES]
    """
    # Convert logits to probabilities
    probs = F.softmax(out, dim=-1)  # Shape: [MODELS, BATCH, NUM_CLASSES]
    probs = probs.transpose(0, 1)  # Shape: [BATCH, MODELS, NUM_CLASSES]

    return divdis_loss_forward_impl(probs, mode=mode, reduction=reduction)


def get_metrics_mapping(oom_safe=False):
    metrics = tuple([
        tuple(["kl", kl_divergence]),
        tuple(["var", div_var]),
        tuple(["std", div_std]),
        tuple(["max_var", max_prob_var])
    ])
    if not oom_safe:
        metrics = tuple(
            [
                tuple(["dis", dis])
            ] + list(metrics)
        )
    return metrics


class A2DLoss(torch.nn.Module):
    def __init__(self, heads, dbat_loss_type='v1', reduction="mean"):
        super().__init__()
        self.heads = heads
        self.dbat_loss_type = dbat_loss_type
        self.reduction = reduction

    # input has shape [batch_size, heads * classes]
    def forward(self, logits):
        logits_chunked = torch.chunk(logits, self.heads, dim=-1)
        probs = torch.stack(logits_chunked, dim=0).softmax(-1)
        m_idx = torch.randint(0, self.heads, (1,)).item()
        # shape [models, batch, classes]
        return a2d_loss_impl(
            probs,
            m_idx,
            dbat_loss_type=self.dbat_loss_type,
            reduction=self.reduction
        )


# based on https://github.com/mpagli/Agree-to-Disagree/blob/d8859164025421e137dca8226ef3b10859bc276c/src/main.py#L92
def a2d_loss_impl(probs, m_idx, dbat_loss_type='v1', reduction='mean'):

    if dbat_loss_type == 'v1':
        adv_loss = []

        p_1_s, indices = [], []

        for i, p_1 in enumerate(probs):
            if i == m_idx:
                continue
            p_1, idx = p_1.max(dim=1)
            p_1_s.append(p_1)
            indices.append(idx)

        p_2 = probs[m_idx]

        # probs for classes predicted by each other model
        p_2_s = [p_2[torch.arange(len(p_2)), max_idx] for max_idx in indices]

        for i in range(len(p_1_s)):
            al = (- torch.log(p_1_s[i] * (1-p_2_s[i]) + p_2_s[i] * (1-p_1_s[i]) + EPS))
            if reduction == 'mean':
                al = al.mean()
            else:
                assert reduction == 'none'

            adv_loss.append(al)

    else:
        raise NotImplementedError("v2 dbat is not implemented yet")

    if reduction == "none":
        agg_func = func_for_dim(torch.mean, 0)
    else:
        assert reduction == "mean"
        agg_func = torch.mean
    return aggregate_tensors_by_func(adv_loss, func=agg_func)


def are_probs(logits):
    if (
            logits.min() >= 0
        and
            logits.max() <= 1
    ):
        return True
    return False


def get_probs(logits):
    if are_probs(logits):
        probs = logits
    else:
        probs = F.softmax(logits, dim=-1)
    return probs


def record_diversity(
    res,
    outputs,
    stacked_outputs,
    metrics_mappings,
    labels=None,
    name_prefix="",
    detailed_results=None
):

    # metrics_mappings is a tuple of tuples:
    # ((name_1, func_1), ... (name_k, func_k))
    for metric_tuple in metrics_mappings:

        metric_name = metric_tuple[0]
        metric_key = name_prefix + metric_name
        compute_metric = metric_tuple[1]
        if metric_name not in res:
            res[metric_key] = 0
        if metric_name in PER_SAMPLE_METRIC_NAMES:
            value = compute_metric(stacked_outputs)
        elif metric_name == "div_ortega":
            assert labels is not None
            value = compute_metric(stacked_outputs, labels).item()
        elif metric_name in STACKED_INPUTS_METRIC_NAMES:
            value = compute_metric(stacked_outputs).item()
        else:
            value = aggregate_tensors_by_func(
                apply_pairwise(outputs, compute_metric)
            ).item()

        if not torch.is_tensor(value):
            res[metric_key] += value

        if detailed_results is not None:
            if metric_key in PER_SAMPLE_METRIC_NAMES:
                if metric_key not in detailed_results:
                    detailed_results[metric_key] = []

                if metric_key in detailed_results:
                    for subvalue in value:
                        detailed_results[metric_key].append(subvalue.item())
