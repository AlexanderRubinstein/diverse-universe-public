import torch
# import itertools
import sys
# import os
# from einops import rearrange
# from torch.distributions.categorical import Categorical
from stuned.utility.utils import get_project_root_path
import torch.nn.functional as F


# local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, get_project_root_path())
# from utility.utils import (
#     NAME_SEP,
#     raise_unknown,
#     check_equal_shape,
#     aggregate_tensors_by_func,
#     get_with_assert,
#     func_for_dim
# )
# from local_datasets.common import get_manifold_projector_dataset
from diverse_universe.local_datasets.utils import (
    # convert_dataset_to_tensor,
    # prepare_for_pickling,
    # record_diversity,
    get_probs
)
# from utility.logger import (
#     make_logger,
#     make_base_estimator_name
# )
# from train_eval.utils import (
#     take_from_2d_tensor,
#     compute_ensemble_output,
#     bootstrap_ensemble_outputs
# )
# from local_models.diverse_vit import (
#     is_diverse_vit_output
# )
# from local_models.model_vs_human import (
#     make_mvh_mapper
# )
# from external_libs.ext_diverse_vit import (
#     losses
# )
# from external_libs.ext_diverse_vit.models import N_HEADS_KEY
sys.path.pop(0)


# ALLOWED_TASK_LOSS_TYPES = ["xce", "bce"]
# LOSS_STATISTICS_NAME = "loss"
# INPUT_GRAD_NAME = "input_grad"
# ORTHOGONAL_GRADIENTS_LOSS = "orthogonal_gradients_loss"
# ON_MANIFOLD_LOSS = "on_manifold_gradients_loss"
# DIV_VIT_LOSS = "diverse_vit_gradients_loss"
# TASK_LOSS_WEIGHT_KEY = "task_loss_weight"
# GRADIENT_BASED_LOSSES_KEY = "gradient_based_losses"
# GRADIENT_BASED_LOSSES_WEIGHTS_KEY = f"{GRADIENT_BASED_LOSSES_KEY}_weights"
# DIVERSE_GRADIENTS_LOSS_KEY = "diverse_gradients_loss"
# DIVDIS_LOSS_KEY = "divdis"
# LAMBDA_KEY = "lambda"
# EPS = 1e-9
# MAX_MODELS_WITHOUT_OOM = 5


# class DiverseGradientsLoss(torch.nn.Module):
#     def __init__(
#         self,
#         task_loss,
#         gradient_based_losses,
#         task_loss_weight=1.0,
#         gradient_based_losses_weights=None,
#         use_max_pred=False,
#         logger=make_logger
#     ):
#         super(DiverseGradientsLoss, self).__init__()
#         self.logger = logger
#         self.task_loss = task_loss
#         # TODO(Alex | 28.07.2022) Use typing instead
#         assert isinstance(gradient_based_losses, list)
#         if gradient_based_losses_weights:
#             assert isinstance(gradient_based_losses_weights, list)
#             assert len(gradient_based_losses_weights) \
#                 == len(gradient_based_losses)
#             assert min(gradient_based_losses_weights) > 0, \
#                 "Expected positive weights"
#             self.gradient_based_losses_weights \
#                 = gradient_based_losses_weights
#         else:
#             self.gradient_based_losses_weights = [
#                 1.0 for _ in range(len(gradient_based_losses))
#             ]
#         self.gradient_based_losses = gradient_based_losses
#         self.task_loss_weight = task_loss_weight
#         self.use_max_pred = use_max_pred

#     def _assert_outputs(self, outputs, targets):

#         if is_diverse_vit_output(outputs):
#             assert "inputs" in outputs[1]
#             outputs = [[outputs[1]["inputs"][0], outputs[0]]]

#         for output in outputs:
#             assert len(output) == 2
#             is_grad_required = output[0].requires_grad
#             assert output[1].shape == targets.shape
#         return is_grad_required

#     def _accumulate_task_loss_gradients(self, outputs, targets, is_grad_required):

#         diverse_vit_forward_info = None

#         if is_diverse_vit_output(outputs):
#             diverse_vit_forward_info = outputs[1]
#             assert "inputs" in diverse_vit_forward_info
#             assert "outputs" in diverse_vit_forward_info
#             assert N_HEADS_KEY in diverse_vit_forward_info
#             outputs = [[diverse_vit_forward_info["inputs"][0], outputs[0]]]

#         input_gradients = []
#         inputs = []
#         total_task_loss = torch.Tensor([0]).to(targets.device)
#         for input, output in outputs:
#             current_task_loss_value = self.task_loss(output, targets)
#             total_task_loss += current_task_loss_value

#             if len(self.gradient_based_losses) > 0 and is_grad_required:
#                 inputs.append(input)
#                 if self.use_max_pred:
#                     differentiable_value = output.max()
#                 else:
#                     differentiable_value = current_task_loss_value
#                 if diverse_vit_forward_info is None:
#                     current_task_loss_grad = torch.autograd.grad(
#                         differentiable_value,
#                         input,
#                         create_graph=True
#                     )[0]
#                     assert current_task_loss_grad is not None
#                     assert len(current_task_loss_grad.shape) >= 2
#                     input_gradients.append(
#                         torch.flatten(current_task_loss_grad, start_dim=1)
#                     )
#                 else:
#                     assert len(outputs) == 1
#                     input_gradients = losses.accumulate_grads(
#                         differentiable_value,
#                         diverse_vit_forward_info["inputs"],
#                         diverse_vit_forward_info["outputs"],
#                         diverse_vit_forward_info[N_HEADS_KEY],
#                         normalize_grads='per_token'
#                     )

#         total_task_loss /= len(outputs)

#         return inputs, input_gradients, total_task_loss

#     def _prepare_for_pickling(self):
#         self.logger = None
#         for loss in self.gradient_based_losses:
#             prepare_for_pickling(loss)

#     def forward(self, outputs, targets):
#         is_grad_required = self._assert_outputs(outputs, targets)
#         inputs, input_gradients, total_task_loss \
#             = self._accumulate_task_loss_gradients(
#                 outputs,
#                 targets,
#                 is_grad_required
#             )

#         total_task_loss *= self.task_loss_weight

#         loss_info = {
#             "{}_task".format(LOSS_STATISTICS_NAME): total_task_loss.item()
#         }

#         if is_diverse_vit_output(outputs):
#             gradients_info = {}
#         else:
#             gradients_info = {
#                 make_base_estimator_name(base_estimator_id):
#                     input_gradient.detach().cpu()
#                         for base_estimator_id, input_gradient
#                             in enumerate(input_gradients)
#             }

#         if input_gradients == []:
#             # gradient-based losses are not computed when they were not provided
#             # or during validation
#             assert (
#                     len(self.gradient_based_losses) == 0
#                 or
#                     is_grad_required == False
#             )
#         else:
#             for weight, gradient_based_loss in zip(
#                 self.gradient_based_losses_weights,
#                 self.gradient_based_losses
#             ):
#                 current_gradient_based_loss_value \
#                     = gradient_based_loss(inputs, input_gradients)
#                 loss_info[gradient_based_loss.loss_name_for_stats] \
#                     = current_gradient_based_loss_value.item()
#                 total_task_loss += weight * current_gradient_based_loss_value

#         loss_info[LOSS_STATISTICS_NAME] = total_task_loss.item()

#         return total_task_loss, loss_info, gradients_info


# def make_name_for_stats(loss_obj):
#     return "{}{}{}".format(
#         LOSS_STATISTICS_NAME,
#         NAME_SEP,
#         str(loss_obj).strip("()")
#     )


# class NamedLoss(torch.nn.Module):

#     def __init__(self):
#         super(NamedLoss, self).__init__()
#         self.loss_name_for_stats = make_name_for_stats(self)


# class DiverseVitGradientsLoss(NamedLoss):

#     def __init__(self):
#         super(DiverseVitGradientsLoss, self).__init__()

#     def forward(self, inputs, input_gradients):
#         assert len(input_gradients) > 0
#         # check for consistency with diverse-vit gradients shape
#         assert len(input_gradients[0].shape) == 3
#         return losses.aggregate_grads(input_gradients)


# class OrthogonalGradientsLoss(NamedLoss):
#     """
#     input: [grad_i]_{i=1}^{n} - list of target loss gradients
#         w.r.t. model input
#     output: \sum_{i}\sum_{j<i} cos^2(grad_i, grad_j) - orthogonality loss
#     """
#     def __init__(self, binary_func="cos_pow_2"):

#         super(OrthogonalGradientsLoss, self).__init__()

#         if binary_func == "cos_pow_2":
#             self.cos = torch.nn.CosineSimilarity(dim=-1)
#             self.func = self._cos_pow_2
#         elif binary_func == "dot":
#             self.func = self._dot
#         else:
#             raise_unknown("binary_func", binary_func, "OrthogonalGradientsLoss")

#     def _dot(self, x, y):
#         return torch.abs(torch.dot(x.view(-1), y.view(-1)))

#     def _cos_pow_2(self, x, y):

#         return torch.pow(
#             self.cos(x, y),
#             2
#         )

#     def forward(self, inputs, input_gradients):

#         check_input_gradients(input_gradients)
#         return aggregate_tensors_by_func(
#             apply_pairwise(
#                 input_gradients,
#                 self.func
#             )
#         )


# class OnManifoldGradientsLoss(NamedLoss):
#     """
#     input:
#         [x_i]_{i=1}^{n} - list of model inputs;
#         [grad_i]_{i=1}^{n} - list of target loss gradients w.r.t. model input.
#     output: \sum_{i}(proj(x_i, grad_i) - grad_i)^2 - on-manifold loss
#         proj(x_i, grad_i) - projection of grad_i on the manifold
#             when it originates in x_i
#     """
#     def __init__(self, projector):
#         super(OnManifoldGradientsLoss, self).__init__()
#         self.projector = projector

#     def _prepare_for_pickling(self):
#         prepare_for_pickling(self.projector)

#     def forward(self, inputs, input_gradients):

#         check_input_gradients(input_gradients)
#         return aggregate_tensors_by_func(
#             [
#                 torch.pow(
#                     torch.subtract(
#                         self.projector(
#                             input,
#                             input_gradient
#                         ),
#                         input_gradient
#                     ),
#                     2
#                 )
#                     for input, input_gradient
#                         in zip(inputs, input_gradients)
#             ]
#         )


# # copied from: https://github.com/yoonholee/DivDis/blob/main/divdis.py#L6
# def to_probs(logits, heads):
#     """
#     Converts logits to probabilities.
#     Input must have shape [batch_size, heads * classes].
#     Output will have shape [batch_size, heads, classes].
#     """

#     B, N = logits.shape
#     if N == heads:  # Binary classification; each head outputs a single scalar.
#         preds = logits.sigmoid().unsqueeze(-1)
#         probs = torch.cat([preds, 1 - preds], dim=-1)
#     else:
#         logits_chunked = torch.chunk(logits, heads, dim=-1)
#         probs = torch.stack(logits_chunked, dim=1).softmax(-1)
#     B, H, D = probs.shape
#     assert H == heads
#     return probs


# # copied from: https://github.com/yoonholee/DivDis/blob/main/divdis.py#L46
# class DivDisLoss(torch.nn.Module):
#     """Computes pairwise repulsion losses for DivDis.

#     Args:
#         logits (torch.Tensor): Input logits with shape [BATCH_SIZE, HEADS * DIM].
#         heads (int): Number of heads.
#         mode (str): DIVE loss mode. One of {pair_mi, total_correlation, pair_l1}.
#     """

#     def __init__(self, heads, mode="mi", reduction="mean"):
#         super().__init__()
#         self.heads = heads
#         self.mode = mode
#         self.reduction = reduction

#     def forward(self, logits):
#         heads, mode, reduction = self.heads, self.mode, self.reduction
#         probs = to_probs(logits, heads)
#         return divdis_loss_forward_impl(probs, mode=mode, reduction=reduction)

#         # if mode == "mi":  # This was used in the paper
#         #     marginal_p = probs.mean(dim=0)  # H, D
#         #     marginal_p = torch.einsum(
#         #         "hd,ge->hgde", marginal_p, marginal_p
#         #     )  # H, H, D, D
#         #     marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # H^2, D^2

#         #     joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(
#         #         dim=0
#         #     )  # H, H, D, D
#         #     joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # H^2, D^2

#         #     # Compute pairwise mutual information = KL(P_XY | P_X x P_Y)
#         #     # Equivalent to: F.kl_div(marginal_p.log(), joint_p, reduction="none")
#         #     kl_computed = joint_p * (joint_p.log() - marginal_p.log())
#         #     kl_computed = kl_computed.sum(dim=-1)
#         #     kl_grid = rearrange(kl_computed, "(h g) -> h g", h=heads)
#         #     repulsion_grid = -kl_grid
#         # elif mode == "l1":
#         #     dists = (probs.unsqueeze(1) - probs.unsqueeze(2)).abs()
#         #     dists = dists.sum(dim=-1).mean(dim=0)
#         #     repulsion_grid = dists
#         # else:
#         #     raise ValueError(f"{mode=} not implemented!")

#         # if reduction == "mean":  # This was used in the paper
#         #     repulsion_grid = torch.triu(repulsion_grid, diagonal=1)
#         #     repulsions = repulsion_grid[repulsion_grid.nonzero(as_tuple=True)]
#         #     repulsion_loss = -repulsions.mean()
#         # elif reduction == "min_each":
#         #     repulsion_grid = torch.triu(repulsion_grid, diagonal=1) + torch.tril(
#         #         repulsion_grid, diagonal=-1
#         #     )
#         #     rows = [r for r in repulsion_grid]
#         #     row_mins = [row[row.nonzero(as_tuple=True)].min() for row in rows]
#         #     repulsion_loss = -torch.stack(row_mins).mean()
#         # else:
#         #     raise ValueError(f"{reduction=} not implemented!")

#         # return repulsion_loss


# def divdis_loss_forward_impl(probs, mode="mi", reduction="mean"):
#     # input has shape [batch_size, heads, classes]
#     # probs = to_probs(logits, heads)
#     heads = probs.shape[1]

#     if mode == "mi":  # This was used in the paper
#         marginal_p = probs.mean(dim=0)  # H, D
#         marginal_p = torch.einsum(
#             "hd,ge->hgde", marginal_p, marginal_p
#         )  # H, H, D, D
#         marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # H^2, D^2

#         joint_p = torch.einsum("bhd,bge->bhgde", probs, probs).mean(
#             dim=0
#         )  # H, H, D, D
#         joint_p = rearrange(joint_p, "h g d e -> (h g) (d e)")  # H^2, D^2

#         # Compute pairwise mutual information = KL(P_XY | P_X x P_Y)
#         # Equivalent to: F.kl_div(marginal_p.log(), joint_p, reduction="none")
#         kl_computed = joint_p * (joint_p.log() - marginal_p.log() + EPS)
#         kl_computed = kl_computed.sum(dim=-1)
#         kl_grid = rearrange(kl_computed, "(h g) -> h g", h=heads)
#         repulsion_grid = -kl_grid
#     elif mode == "l1":
#         dists = (probs.unsqueeze(1) - probs.unsqueeze(2)).abs()
#         dists = dists.sum(dim=-1).mean(dim=0)
#         repulsion_grid = dists
#     else:
#         raise ValueError(f"{mode=} not implemented!")

#     if reduction == "mean":  # This was used in the paper
#         repulsion_grid = torch.triu(repulsion_grid, diagonal=1)
#         repulsions = repulsion_grid[repulsion_grid.nonzero(as_tuple=True)]
#         repulsion_loss = -repulsions.mean()
#     elif reduction == "min_each":
#         repulsion_grid = torch.triu(repulsion_grid, diagonal=1) + torch.tril(
#             repulsion_grid, diagonal=-1
#         )
#         rows = [r for r in repulsion_grid]
#         row_mins = [row[row.nonzero(as_tuple=True)].min() for row in rows]
#         repulsion_loss = -torch.stack(row_mins).mean()
#     else:
#         raise ValueError(f"{reduction=} not implemented!")

#     return repulsion_loss


# def focal_modifier(output, target, gamma):
#     output = get_probs(output)
#     modifier = torch.pow(
#         (1 - take_from_2d_tensor(output, target, dim=-1)),
#         gamma
#     )
#     return modifier


# def budget_modifier(outputs, target):
#     with torch.no_grad():
#         ensemble_output = compute_ensemble_output(outputs)

#         unreduced_ce = F.cross_entropy(
#             ensemble_output,
#             target,
#             reduction='none'
#         )
#         divider = torch.pow(unreduced_ce.mean(0), 2)
#         return unreduced_ce / divider


# class DivDisLossWrapper(torch.nn.Module):

#     def __init__(
#         self,
#         task_loss,
#         weight,
#         mode="mi",
#         reduction="mean",
#         mapper=None,
#         loss_type="divdis",
#         use_always_labeled=False,
#         modifier=None,
#         gamma=2.0,
#         disagree_after_epoch=0,
#         manual_lambda=1.0,
#         disagree_below_threshold=None,
#         reg_mode=None,
#         reg_weight=None
#     ):
#         super().__init__()
#         self.repulsion_loss = None
#         self.mode = mode
#         self.reduction = reduction
#         self.task_loss = task_loss
#         self.weight = weight
#         self.loss_type = loss_type
#         if mapper == "mvh":
#             self.mapper = make_mvh_mapper()
#         else:
#             assert mapper is None, "Only mvh mapper is supported"
#             self.mapper = None
#         self.log_this_batch = True
#         self.use_always_labeled = use_always_labeled
#         self.modifier = modifier
#         self.gamma = gamma
#         self.epoch = 0
#         self.disagree_after_epoch = disagree_after_epoch
#         self.manual_lambda = manual_lambda
#         self.disagree_below_threshold = disagree_below_threshold
#         self.reg_mode = reg_mode
#         self.reg_weight = reg_weight

#         if self.reg_mode is not None:
#             assert self.reg_weight is not None

#     def increase_epoch(self):
#         self.epoch += 1

#     def compute_modifier(self, outputs, targets):

#         if self.modifier == "focal":
#             # (1 - pt) ** 2
#             return aggregate_tensors_by_func(
#                 [
#                     focal_modifier(output, targets, self.gamma)
#                         for output in outputs
#                 ]
#             )
#         elif self.modifier == "budget":
#             return budget_modifier(outputs, targets)
#         else:
#             assert self.modifier is None
#             return 1.0

#     def forward(self, outputs, targets, unlabeled_outputs=None):

#         def zero_if_none(value):
#             return (
#                 value.item()
#                     if value is not None
#                     else 0
#             )

#         def get_repulsion_loss(outputs, unlabeled_outputs, targets_values):
#             if unlabeled_outputs is None:
#                 assert not outputs[0][1].requires_grad, \
#                     "No unlabeled batch was provided during training"
#                 repulsion_loss = None
#                 modifier = None
#             else:
#                 n_heads = len(unlabeled_outputs)
#                 if self.repulsion_loss is None:

#                     if self.loss_type == "divdis":
#                         assert self.modifier != "budget", \
#                             "Budget modifier is not supported for DivDisLoss"
#                         self.repulsion_loss = DivDisLoss(
#                             n_heads,
#                             self.mode,
#                             self.reduction
#                         )
#                     else:
#                         assert self.loss_type == "a2d"
#                         reduction = "mean"
#                         if self.modifier == "budget":
#                             reduction = "none"
#                         self.repulsion_loss = A2DLoss(
#                             n_heads,
#                             reduction=reduction
#                         )
#                 else:
#                     self.repulsion_loss.heads = n_heads

#                 if self.use_always_labeled:
#                     modifier = self.compute_modifier(
#                         unlabeled_outputs,
#                         targets_values
#                     )
#                 else:
#                     assert self.modifier is None
#                     modifier = 1.0

#                 if self.mapper is not None:
#                     unlabeled_outputs = [self.mapper(output) for output in unlabeled_outputs]

#                 # [batch, n * classes]
#                 unlabeled_outputs_cat = torch.cat(
#                     unlabeled_outputs,
#                     axis=-1
#                 )

#                 repulsion_loss = self.repulsion_loss(unlabeled_outputs_cat)
#             return repulsion_loss, modifier, unlabeled_outputs

#         if self.use_always_labeled:
#             assert unlabeled_outputs is None
#             unlabeled_outputs = outputs

#         cur_n_heads = len(outputs)
#         if cur_n_heads > MAX_MODELS_WITHOUT_OOM:
#             metrics_mappings = OOM_SAFE_METRICS
#         else:
#             metrics_mappings = METRICS

#         outputs = bootstrap_ensemble_outputs(outputs)
#         targets_values = targets.max(-1).indices
#         for output in outputs:
#             assert not torch.isnan(output).any(), "NaNs in outputs"
#         if unlabeled_outputs is not None:
#             unlabeled_outputs = bootstrap_ensemble_outputs(unlabeled_outputs)

#         repulsion_loss, modifier, reg_loss = None, None, None

#         if self.weight > 0 and self.epoch + 1 > self.disagree_after_epoch:
#             if self.disagree_below_threshold is not None:
#                 assert self.modifier is None, \
#                     "Can't use modifier with disagree_below_threshold"
#                 assert self.weight < 1, \
#                     "Can't have lambda == 1 with disagree_below_threshold"
#                 assert self.use_always_labeled

#                 masks = [
#                     (
#                             take_from_2d_tensor(
#                                 get_probs(output),
#                                 targets_values,
#                                 dim=-1
#                             )
#                         >
#                             self.disagree_below_threshold
#                     )
#                         for output
#                         in outputs
#                 ]
#                 # take samples which are low prob for all models
#                 mask = torch.stack(masks).min(0).values
#                 unlabeled_outputs = [
#                     output[~mask, ...]
#                         for output
#                         in outputs
#                 ]
#                 outputs = [output[mask, ...] for output in outputs]
#                 targets = targets[mask, ...]
#             if (
#                     unlabeled_outputs is not None
#                 and
#                     len(unlabeled_outputs) > 0
#                 and
#                     len(unlabeled_outputs[0]) > 0
#             ):
#                 repulsion_loss, modifier, unlabeled_outputs = get_repulsion_loss(
#                     outputs,
#                     unlabeled_outputs,
#                     targets_values
#                 )

#                 reg_loss = get_regularizer(
#                     self.reg_mode,
#                     outputs,
#                     unlabeled_outputs
#                 )

#         if repulsion_loss is not None:
#             assert not torch.isnan(repulsion_loss).any(), "NaNs in repulsion_loss"

#         task_loss_value = torch.Tensor([0])[0]
#         total_loss = task_loss_value.to(targets.device)
#         if self.weight < 1:
#             if len(outputs) > 0 and len(outputs[0]) > 0:
#                 task_loss_value = aggregate_tensors_by_func(
#                     [self.task_loss(output, targets) for output in outputs]
#                 )

#                 total_loss = (1 - self.weight) * task_loss_value
#         else:
#             assert self.weight == 1
#             assert self.disagree_after_epoch == 0, \
#                 "When lambda is 1 disagreement should start from the first epoch"

#         if repulsion_loss is not None:

#             repulsion_loss *= modifier

#             if len(repulsion_loss.shape) > 0 and repulsion_loss.shape[0] > 1:
#                 repulsion_loss = repulsion_loss.mean()

#             total_loss += self.manual_lambda * self.weight * repulsion_loss

#         if reg_loss is not None:
#             total_loss += self.reg_weight * self.weight * reg_loss

#         loss_info = {
#             "task_loss": task_loss_value.item(),
#             "repulsion_loss": zero_if_none(repulsion_loss),
#             "regularizer_loss": zero_if_none(reg_loss),
#             "total_loss": total_loss.item()
#         }

#         if self.log_this_batch:
#             record_diversity(
#                 loss_info,
#                 outputs,
#                 torch.stack(outputs),
#                 metrics_mappings=metrics_mappings,
#                 name_prefix="ID_loss_"
#             )
#             if unlabeled_outputs is not None and not self.use_always_labeled:

#                 record_diversity(
#                     loss_info,
#                     unlabeled_outputs,
#                     torch.stack(unlabeled_outputs),
#                     metrics_mappings=metrics_mappings,
#                     name_prefix="OOD_loss_"
#                 )
#         gradients_info = {}
#         return total_loss, loss_info, gradients_info


# # inspired by this: https://github.com/yoonholee/DivDis/blob/b9de1a637949594054240254f667063788ee1573/subpopulation/train.py#L197-L220
# def get_regularizer(reg_mode, outputs, unlabeled_outputs):

#     def chunk(outputs, heads):
#         outputs_cat = torch.cat(
#             outputs,
#             axis=-1
#         )
#         chunked = torch.chunk(outputs_cat, heads, dim=-1)
#         return chunked

#     if reg_mode is None:
#         return None

#     assert reg_mode == "kl_backward"
#     heads = len(outputs)

#     yhat_chunked = chunk(outputs, heads)
#     yhat_unlabeled_chunked = chunk(unlabeled_outputs, heads)

#     preds = torch.stack(yhat_unlabeled_chunked).softmax(-1)

#     # TODO(Alex |09.05.2024): avoid chunking and then stacking
#     avg_preds_source = (
#         torch.stack(yhat_chunked).softmax(-1).mean([0, 1]).detach()
#     )
#     avg_preds_target = preds.mean(1)
#     dist_source = Categorical(probs=avg_preds_source)
#     dist_target = Categorical(probs=avg_preds_target)
#     if reg_mode in ["kl_forward", "kl_ratio_f"]:
#         kl = torch.distributions.kl.kl_divergence(dist_source, dist_target)
#     elif reg_mode in ["kl_backward", "kl_ratio_b"]:
#         kl = torch.distributions.kl.kl_divergence(dist_target, dist_source)
#     reg_loss = kl.mean()
#     return reg_loss


# def make_divdis_loss(divdis_loss_config, logger):
#     task_loss = make_criterion(
#         get_with_assert(divdis_loss_config, "task_loss"),
#         logger=logger
#     )
#     repulsion_loss_type = divdis_loss_config.get("loss_type", "divdis")
#     logger.log(
#         "Using repulsion_loss \"{}\" for DivDisLoss".format(repulsion_loss_type)
#     )
#     mapper = divdis_loss_config.get("mapper")
#     reg_mode = divdis_loss_config.get("reg_mode")
#     if reg_mode is not None:
#         logger.log(
#             f"Using regularizer {reg_mode}"
#         )
#     if mapper is not None:
#         logger.log("Using mapper \"{}\" for DivDisLoss".format(mapper))
#     return DivDisLossWrapper(
#         task_loss,
#         get_with_assert(divdis_loss_config, "lambda"),
#         divdis_loss_config.get("mode", "mi"),
#         divdis_loss_config.get("reduction", "mean"),
#         mapper=mapper,
#         loss_type=repulsion_loss_type,
#         use_always_labeled=divdis_loss_config.get("use_always_labeled", False),
#         modifier=divdis_loss_config.get("modifier"),
#         gamma=divdis_loss_config.get("gamma", 2.0),
#         disagree_after_epoch=divdis_loss_config.get("disagree_after_epoch", 0),
#         manual_lambda=divdis_loss_config.get("manual_lambda", 1.0),
#         disagree_below_threshold=divdis_loss_config.get(
#             "disagree_below_threshold",
#             None
#         ),
#         reg_mode=reg_mode,
#         reg_weight=divdis_loss_config.get("reg_weight")
#     )


# class Projector:
#     def __init__(self, logger=make_logger(), device="cpu"):
#         self.logger = logger
#         self.device = device
#         self.fitted = False

#     def _prepare_for_pickling(self):
#         self.logger = None

#     def _do_fit(self, data):
#         pass

#     def fit(self, data):
#         self._do_fit(data)
#         self.fitted = True

#     def _do_call(self, x, grad_x):
#         pass

#     def __call__(self, x, grad_x):
#         if not self.fitted:
#             raise Exception("Projector was used before fitting to any data!")
#         return self._do_call(x, grad_x)


# class ProjectorPCA(Projector):
#     def __init__(self, n_components, logger=make_logger(), device="cpu"):

#         super(ProjectorPCA, self).__init__(logger=logger, device=device)

#         from sklearn.decomposition import PCA # slow

#         self.pca = PCA(n_components=n_components)

#     def _do_fit(self, dataset):
#         data = convert_dataset_to_tensor(dataset)[0]
#         assert len(data.shape) == 2
#         self.pca.fit(data)

#         # rows - principal vectors
#         self.V = torch.Tensor(self.pca.components_).to(self.device)
#         if len(self.V.shape) == 1:
#             self.V = self.V[None, ...]

#     def _do_call(self, x, grad_x):
#         # <x> is not used as we just require <grad_x> to be
#         # within the hyperplane spanned by principal axes

#         coordinates_in_principal_basis = torch.einsum(
#             'ij,bj->bi',
#             self.V,
#             grad_x
#         )

#         return torch.einsum(
#             'ji,bi->bj',
#             self.V.T,
#             coordinates_in_principal_basis
#         )

# # TODO(Alex | 17.01.2024): Better use the one from utils instead
# def apply_pairwise(iterable, func):
#     pairs = itertools.combinations(iterable, 2)
#     res = []
#     for a, b in pairs:
#         res.append(func(a, b))
#     return res


# def check_input_gradients(input_gradients):
#     assert input_gradients
#     shape = input_gradients[0].shape
#     assert len(shape) > 0 and len(shape) <=2
#     check_equal_shape(input_gradients)


# def make_criterion(
#     criterion_config,
#     cache_path=None,
#     logger=make_logger(),
#     device="cpu"
# ):
#     criterion_type = criterion_config["type"]
#     logger.log("Making criterion \"{}\"..".format(criterion_type))
#     if criterion_type == "bce":
#         criterion = torch.nn.BCELoss()
#     elif criterion_type == "xce":
#         criterion = torch.nn.CrossEntropyLoss()
#     elif criterion_type == DIVDIS_LOSS_KEY:
#         criterion = make_divdis_loss(criterion_config[criterion_type], logger)
#     elif criterion_type == DIVERSE_GRADIENTS_LOSS_KEY:
#         criterion = make_diverse_gradients_loss(
#             criterion_config[criterion_type],
#             cache_path,
#             logger,
#             device
#         )
#     else:
#         raise_unknown("criterion", criterion_type, "criterion config")

#     criterion.smoothing_eps = criterion_config.get("smoothing_eps")

#     return criterion


# def make_diverse_gradients_loss(
#     criterion_subconfig,
#     cache_path=None,
#     logger=make_logger(),
#     device="cpu"
# ):
#     task_loss_config = criterion_subconfig["task_loss"]
#     assert task_loss_config["type"] in ALLOWED_TASK_LOSS_TYPES
#     task_loss = make_criterion(task_loss_config, logger)

#     gradient_based_losses = []
#     for gradient_based_loss_type \
#         in criterion_subconfig[GRADIENT_BASED_LOSSES_KEY]:

#         specific_gradient_based_loss_config = criterion_subconfig.get(
#             gradient_based_loss_type,
#             {}
#         )

#         if gradient_based_loss_type == ORTHOGONAL_GRADIENTS_LOSS:

#             gradient_based_losses.append(
#                 OrthogonalGradientsLoss(
#                     binary_func=specific_gradient_based_loss_config.get(
#                         "binary_func",
#                         "cos_pow_2"
#                     )
#                 )
#             )

#         elif gradient_based_loss_type == ON_MANIFOLD_LOSS:
#             projector = make_projector(
#                 specific_gradient_based_loss_config["projector"],
#                 cache_path,
#                 logger,
#                 device
#             )
#             gradient_based_losses.append(OnManifoldGradientsLoss(
#                 projector=projector
#             ))
#         elif gradient_based_loss_type == DIV_VIT_LOSS:
#             gradient_based_losses.append(DiverseVitGradientsLoss())
#         else:
#             raise_unknown(
#                 "gradient_based_loss type",
#                 gradient_based_loss_type,
#                 f"{DIVERSE_GRADIENTS_LOSS_KEY} config"
#             )

#     for gradient_based_loss in gradient_based_losses:
#         assert isinstance(gradient_based_loss, NamedLoss)

#     return DiverseGradientsLoss(
#         task_loss,
#         gradient_based_losses,
#         task_loss_weight=criterion_subconfig.get(TASK_LOSS_WEIGHT_KEY, 1.0),
#         gradient_based_losses_weights=criterion_subconfig[
#             GRADIENT_BASED_LOSSES_WEIGHTS_KEY
#         ],
#         use_max_pred=criterion_subconfig.get("use_max_pred", False),
#         logger=logger
#     )


# def make_projector(
#     projector_config,
#     cache_path=None,
#     logger=make_logger(),
#     device="cpu"
# ):
#     projector_type = projector_config["type"]
#     logger.log("Making projector \"{}\"..".format(projector_type))
#     if projector_type == "PCA":
#         projector = ProjectorPCA(
#             n_components=projector_config["latent_dim"],
#             logger=logger,
#             device=device
#         )
#     else:
#         raise_unknown(
#             "projector_type",
#             projector_type,
#             "projector_config"
#         )
#     projector_data_config = projector_config["unlabeled_data"]
#     projector_dataset = get_manifold_projector_dataset(
#         projector_data_config,
#         cache_path,
#         logger
#     )
#     logger.log(
#         "Fitting {} projector for data {}..".format(
#             projector_type,
#             projector_data_config["type"]
#         )
#     )
#     projector.fit(projector_dataset)
#     return projector


# def check_criterion(criterion, expected_criterion):
#     pass
#     # if expected_criterion == "bce":
#     #     assert \
#     #         isinstance(criterion, torch.nn.BCELoss), \
#     #         "Expected torch.nn.BCELoss"
#     # elif expected_criterion == "xce":
#     #     assert \
#     #         isinstance(criterion, torch.nn.CrossEntropyLoss), \
#     #         "Expected torch.nn.CrossEntropyLoss"
#     # elif expected_criterion == DIVERSE_GRADIENTS_LOSS_KEY:
#     #     assert \
#     #         isinstance(criterion, DiverseGradientsLoss), \
#     #         "Expected DiverseGradientsLoss"
#     # else:
#     #     raise_unknown(
#     #         "expected_criterion",
#     #         expected_criterion,
#     #         "checkpoint"
#     #     )


# def requires_input_gradient(criterion):
#     if isinstance(criterion, DiverseGradientsLoss):
#         return True
#     return False


# # maybe some of the below are not needed
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import webbrowser
# import torch.distributions as dist
# # import ot
# from torchmetrics import JaccardIndex
# # from torch.utils.data import DataLoader
# from einops import rearrange

# # from utils.simple_io import *
# # from tensorboard import program
# # from torch.utils.tensorboard import SummaryWriter


# ##### Luca's metrics from: https://github.com/lucascimeca/bse_div/blob/97e97b6d41e4dea7dfb854631282e5506401ff2c/src/utils/pytorch_utils.py#L19
# # div and std from here: https://github.com/lucascimeca/bse_div/blob/97e97b6d41e4dea7dfb854631282e5506401ff2c/src/utils/pytorch_utils.py#L248


# def div_entropy(logits):
#     probs = get_probs(logits)
#     return -(probs * torch.log(probs + EPS)).sum(-1).mean()


# def div_mean_prob(logits):
#     probs = get_probs(logits)
#     return div_mean_logit(probs)


# def div_max_prob(logits):
#     probs = get_probs(logits)
#     return div_max_logit(probs)


# def div_mean_logit(logits):
#     return logits.mean()


# def div_max_logit(logits):
#     return logits.max(-1).values.mean()


def div_different_preds(logits):

    return div_different_preds_per_sample(logits).mean()


def div_continous_unique(logits):

    return div_continous_unique_per_sample(logits).mean()


def div_continous_unique_per_sample(logits):
    probs = get_probs(logits).clone()

    max_by_model = probs.max(0).values
    sum_over_classes = max_by_model.sum(-1)

    # probs_batch_first = probs.transpose(0, 1)
    # num_models, batch_size, num_classes = probs.shape
    # res = []
    # for sample_probs in probs_batch_first:
    #     score = 0
    #     for i in range(num_models):

    #         max_prob_classes = sample_probs.max(-1)
    #         max_prob_models = max_prob_classes.values.max(-1)
    #         m_star = max_prob_models.indices
    #         k_star = max_prob_classes.indices[m_star]
    #         max_prob = max_prob_models.values

    #         score += max_prob
    #         sample_probs[m_star, k_star] = -1
    #     res.append(score)

    return torch.Tensor(sum_over_classes)


def div_different_preds_per_sample(logits):

    probs = get_probs(logits)
    preds = probs.argmax(-1).t()
    res = []

    for per_sample_preds in preds:
        res.append(len(per_sample_preds.unique()))

    return torch.Tensor(res)


# def div_different_preds_per_model(logits):

#     probs = get_probs(logits)
#     preds = probs.argmax(-1)
#     res = []

#     for per_model_preds in preds:
#         res.append(len(per_model_preds.unique()))

#     return torch.Tensor(res).mean()


# # based on ce diversity from here: https://arxiv.org/abs/2110.13786
# def div_ortega(logits, targets):
#     sqrt2 = torch.sqrt(torch.Tensor([2.0])).to(targets.device)
#     probs = F.softmax(logits, dim=-1)
#     num_models, batch_size, num_classes = probs.shape
#     stacked_probs = probs[
#         torch.arange(num_models).repeat_interleave(batch_size),
#         torch.arange(batch_size).repeat(num_models),
#         targets.repeat(num_models)
#     ].reshape(num_models, batch_size)

#     max_across_models = stacked_probs.max(0).values

#     stacked_probs /= sqrt2 * max_across_models
#     var = stacked_probs.var(0).mean()
#     return var


def div_var(outputs):
    return -F.softmax(outputs, dim=-1).var(0).mean()


# def max_prob_var(outputs):

#     probs = F.softmax(outputs, dim=-1)
#     max_across_models = torch.max(probs, dim=-1).indices
#     probs = probs[..., max_across_models]

#     return probs.var(0).mean()


# def div_std(outputs):
#     return -F.softmax(outputs, dim=-1).std(0).mean()


# def kl_divergence(out1, out2):
#     loss = F.kl_div(torch.log(F.softmax(out1, dim=-1) + 1e-8), F.softmax(out2, dim=-1) + 1e-8, reduction='batchmean')
#     return -loss  # negative because we want to maximize KL divergence


# def js_divergence(out1, out2):
#     p1 = F.softmax(out1, dim=-1)
#     p2 = F.softmax(out2, dim=-1)
#     return - 0.5 * (F.kl_div(torch.log(p1 + 1e-8), 0.5 * (p1 + p2), reduction='batchmean') +
#                     F.kl_div(torch.log(p2 + 1e-8), 0.5 * (p1 + p2), reduction='batchmean'))


# def cosine_similarity(out1, out2):
#     p = F.softmax(out1, dim=-1)
#     q = F.softmax(out2, dim=-1)
#     similarity = F.cosine_similarity(p, q, dim=-1)
#     return 1 - similarity.mean()


# def orthogonality_loss(out1, out2):
#     p1 = F.softmax(out1, dim=-1)
#     p2 = F.softmax(out2, dim=-1)

#     # Calculate dot product along the class dimension
#     dot_product = torch.sum(p1 * p2, dim=-1)

#     # We aim to minimize the dot product to enforce orthogonality
#     loss = dot_product.mean()

#     return loss


# def euclidean_distance(out1, out2):
#     p = F.softmax(out1, dim=-1)
#     q = F.softmax(out2, dim=-1)
#     return -F.mse_loss(p, q, reduction='mean').sqrt()


# def jaccard_similarity(out1, out2, k=5):
#     k = min(out1.shape[-1], k)
#     jaccard = JaccardIndex(num_classes=k, task='multiclass').to(out1.device)
#     score = jaccard(out1.argmax(dim=-1), out2.argmax(dim=-1))
#     return score

# # import ot
# # def wasserstein_distance(out1, out2):
# #     p = F.softmax(out1, dim=-1).detach().cpu().numpy()
# #     q = F.softmax(out2, dim=-1).detach().cpu().numpy()
# #     M = ot.dist(p.reshape(-1, 1), q.reshape(-1, 1))
# #     loss = ot.emd2([], [], M)
# #     return -torch.tensor(loss, requires_grad=True)


# def spearman_rank(out1, out2):
#     p = F.softmax(out1, dim=-1)
#     q = F.softmax(out2, dim=-1)
#     rank_p = torch.argsort(torch.argsort(p, dim=-1, descending=True), dim=-1)
#     rank_q = torch.argsort(torch.argsort(q, dim=-1, descending=True), dim=-1)
#     diff = rank_p - rank_q
#     return torch.mean(diff.float() ** 2)


# def reverse_cross_entropy(out1, out2):
#     p = F.softmax(out1, dim=-1)
#     q = F.softmax(out2, dim=-1)
#     ac = F.cross_entropy(p, q.topk(1, 1, True, True).indices.view(-1))
#     bc = F.cross_entropy(q, p.topk(1, 1, True, True).indices.view(-1))
#     return -(ac + bc) / 2


# def sinkhorn_distance(out1, out2, epsilon=0.1, max_iters=100):
#     """
#     Compute the Sinkhorn distance between two discrete probability distributions mu and nu.

#     epsilon: Entropy regularization term.
#     max_iters: Maximum number of iterations for the Sinkhorn algorithm.

#     Returns:
#     sinkhorn_dist: torch.Tensor of shape (batch_size,)
#                     The batched Sinkhorn distances.
#     """

#     # mu, nu: torch.Tensor of shape (batch_size, num_classes).
#     # They represent batched discrete probability distributions.
#     mu = torch.nn.functional.softmax(out1, dim=-1)
#     nu = torch.nn.functional.softmax(out2, dim=-1)

#     # Number of classes
#     C = mu.shape[1]

#     # Cost matrix
#     x = torch.arange(C, dtype=torch.float32).view(-1, 1)
#     y = torch.arange(C, dtype=torch.float32).view(1, -1)
#     M = (x - y) ** 2

#     # Kernel matrix
#     K = torch.exp(-M / epsilon)

#     # Sinkhorn iterations
#     a = torch.ones(C).unsqueeze(0) / C
#     b = a.clone()

#     for _ in range(max_iters):
#         a = mu / (torch.mm(K, b.T).T + 1e-8)
#         b = nu / (torch.mm(K.T, a.T).T + 1e-8)

#     P = a.unsqueeze(2) * K * b.unsqueeze(1)
#     sinkhorn_dist = (P * M).sum(dim=(1, 2))

#     return sinkhorn_dist.mean()


# def dis(out, mode="mi", reduction="mean"):
#     """
#     Calculate the loss to minimize the mutual information between all pairs of models.
#     `out` should have shape [MODELS, BATCH, NUM_CLASSES]
#     """
#     # Convert logits to probabilities
#     probs = F.softmax(out, dim=-1)  # Shape: [MODELS, BATCH, NUM_CLASSES]
#     probs = probs.transpose(0, 1)  # Shape: [BATCH, MODELS, NUM_CLASSES]

#     return divdis_loss_forward_impl(probs, mode=mode, reduction=reduction)


# # def dis(out, mode="mi", reduction="mean"):
# #     """
# #     Calculate the loss to minimize the mutual information between all pairs of models.
# #     `out` should have shape [MODELS, BATCH, NUM_CLASSES]
# #     """
# #     # Convert logits to probabilities
# #     probs = F.softmax(out, dim=-1)  # Shape: [MODELS, BATCH, NUM_CLASSES]
# #     # probs = probs.transpose(0, 1)  # Shape: [BATCH, MODELS, NUM_CLASSES]

# #     # Calculate marginal probabilities across batches
# #     marginal_p = torch.mean(probs, dim=1)  # Shape: [MODELS, NUM_CLASSES]

# #     # Compute marginal joint probabilities
# #     marginal_p = torch.einsum("hd,ge->hgde", marginal_p, marginal_p)  # Shape: [MODELS, MODELS, NUM_CLASSES, NUM_CLASSES]
# #     marginal_p = rearrange(marginal_p, "h g d e -> (h g) (d e)")  # Shape: [MODELS^2, NUM_CLASSES^2]

# #     # Compute joint probabilities
# #     joint_p = torch.einsum("mbd,nbe->mnbde", probs, probs).mean(
# #         dim=2)  # Shape should be [MODEL_NUM, MODEL_NUM, NUM_CLASSES, NUM_CLASSES]
# #     joint_p = rearrange(joint_p, "m n d e -> (m n) (d e)")  # Should be [MODEL_NUM^2, NUM_CLASSES^2]

# #     # Compute KL divergence (mutual information approximation)
# #     kl_computed = joint_p * (joint_p.log() - marginal_p.log() + 1e-9)
# #     kl_computed = kl_computed.sum(dim=-1)  # Summing over all classes

# #     # Reshape and negate to get the final loss grid
# #     kl_grid = rearrange(kl_computed, "(h g) -> h g", h=int(out.shape[0]))
# #     repulsion_grid = -kl_grid

# #     repulsion_grid = torch.triu(repulsion_grid, diagonal=1)
# #     repulsions = repulsion_grid[repulsion_grid.nonzero(as_tuple=True)]
# #     repulsion_loss = -repulsions.mean()
# #     print("OLD")
# #     return repulsion_loss


# def _compute_diff(self, diff_outputs, num_chunks=None, device='cpu'):
#     # a = torch.LongTensor([[0, 0, 1],
#     #                       [0, 1, 0]])
#     #
#     # b1 = torch.FloatTensor([[0, 1, 0],
#     #                         [0, 1, 0]])
#     #
#     # b2 = torch.FloatTensor([[1, 0, 0],
#     #                         [0, 1, 0]])
#     #
#     # c1 = torch.FloatTensor([[1, 0, 0],
#     #                         [1, 0, 0]])
#     #
#     # c2 = torch.FloatTensor([[1, 0, 0],
#     #                         [0, 0, 1]])
#     #
#     # def diff(tensors):
#     #     return torch.stack(tensors).var(0).mean()
#     #
#     # def diff(tensors):
#     #     diversity = []
#     #     for i in range(len(tensors)):
#     #         for j in range(i + 1, len(tensors)):
#     #             diversity.append(F.mse_loss(tensors[i], tensors[j]))
#     #     return sum(diversity) / len(diversity)
#     # def diff(tensors):
#     #     diversity = []
#     #     for i in range(len(tensors)):
#     #         for j in range(i + 1, len(tensors)):
#     #             diversity.append(F.cross_entropy(F.softmax(tensors[i].float(), dim=-1),
#     #                                              tensors[j].topk(1, 1, True, True).indices.view(-1)))
#     #     return sum(diversity) / len(diversity)
#     #
#     # ab1 = diff([a, b1, b2])
#     # ab2 = diff([a, b1, c1])
#     # ac1 = diff([a, c1, c2])
#     # ac2 = diff([a, b2, c2])
#     # print(f"{ab1} {ab2} {ac1} {ac2}")

#     # check if we have outputs by models on OOD data
#     if diff_outputs is not None:

#         if not isinstance(diff_outputs, torch.Tensor):
#             diff_outputs = torch.stack(diff_outputs)

#         # if num_chunks is None:
#         #     num_chunks = diff_outputs.shape[1] // 512

#         # batches = torch.chunk(diff_outputs, num_chunks, dim=1)

#         # losses = 0.
#         # for batch in batches:
#         if 'var' in self.metric or 'std' in self.metric:
#             if 'var' in self.metric:
#                 disentanglement_loss = -F.softmax(diff_outputs.to(device), dim=-1).var(0).mean()
#             else:
#                 disentanglement_loss = -F.softmax(diff_outputs.to(device), dim=-1).std(0).mean()

#         elif 'dis' in self.metric:
#             disentanglement_loss = self.dis(diff_outputs)

#         else:
#             diversity = []
#             for i in range(len(diff_outputs)):
#                 for j in range(i + 1, len(diff_outputs)):
#                     diversity.append(self.diff_measure(diff_outputs[i].to(device), diff_outputs[j].to(device)))
#             disentanglement_loss = (sum(diversity) / len(diversity))

#             # losses += disentanglement_loss
#         # disentanglement_loss = losses / len(batches)
#     else:
#         disentanglement_loss = torch.FloatTensor([0]).mean().to(device)

#     return disentanglement_loss

# OOM_SAFE_METRICS = tuple([
#     tuple(["kl", kl_divergence]),
#     # tuple("js", js_divergence),
#     # tuple("cos", cosine_similarity),
#     # tuple("orth", orthogonality_loss),
#     # tuple("euc", euclidean_distance),
#     # tuple("jac", jaccard_similarity),
#     # tuple("wass", wasserstein_distance),
#     # tuple("spear", spearman_rank),
#     # tuple("rev", reverse_cross_entropy),
#     # tuple("sink", sinkhorn_distance),
#     # tuple(["dis", dis]),
#     tuple(["var", div_var]),
#     tuple(["std", div_std]),
#     tuple(["max_var", max_prob_var])
# ])
# METRICS = tuple(
#     [
#         tuple(["dis", dis])
#     ] + list(OOM_SAFE_METRICS)
# )


# class A2DLoss(torch.nn.Module):
#     def __init__(self, heads, dbat_loss_type='v1', reduction="mean"):
#         super().__init__()
#         self.heads = heads
#         self.dbat_loss_type = dbat_loss_type
#         self.reduction = reduction

#     # input has shape [batch_size, heads * classes]
#     def forward(self, logits):
#         logits_chunked = torch.chunk(logits, self.heads, dim=-1)
#         probs = torch.stack(logits_chunked, dim=0).softmax(-1)
#         m_idx = torch.randint(0, self.heads, (1,)).item()
#         # shape [models, batch, classes]
#         return a2d_loss_impl(
#             probs,
#             m_idx,
#             dbat_loss_type=self.dbat_loss_type,
#             reduction=self.reduction
#         )


# # based on https://github.com/mpagli/Agree-to-Disagree/blob/d8859164025421e137dca8226ef3b10859bc276c/src/main.py#L92
# def a2d_loss_impl(probs, m_idx, dbat_loss_type='v1', reduction='mean'):

#     if dbat_loss_type == 'v1':
#         adv_loss = []

#         p_1_s, indices = [], []

#         for i, p_1 in enumerate(probs):
#             if i == m_idx:
#                 continue
#             p_1, idx = p_1.max(dim=1)
#             p_1_s.append(p_1)
#             indices.append(idx)

#         p_2 = probs[m_idx]

#         # probs for classes predicted by each other model
#         p_2_s = [p_2[torch.arange(len(p_2)), max_idx] for max_idx in indices]

#         for i in range(len(p_1_s)):
#             al = (- torch.log(p_1_s[i] * (1-p_2_s[i]) + p_2_s[i] * (1-p_1_s[i]) + EPS))
#             if reduction == 'mean':
#                 al = al.mean()
#             else:
#                 assert reduction == 'none'

#             adv_loss.append(al)
#     # elif dbat_loss_type == 'v2':
#     #     adv_loss = []
#     #     p_2 = torch.softmax(m(x_tilde), dim=1)
#     #     p_2_1, max_idx = p_2.max(dim=1) # proba of class 1 for m

#     #     with torch.no_grad():
#     #         p_1_s = [torch.softmax(m_(x_tilde), dim=1) for m_ in ensemble[:m_idx]]
#     #         p_1_1_s = [p_1[torch.arange(len(p_1)), max_idx] for p_1 in p_1_s] # probas of class 1 for m_

#     #     for i in range(len(p_1_s)):
#     #         al = (- torch.log(p_1_1_s[i] * (1.0 - p_2_1) + p_2_1 * (1.0 - p_1_1_s[i]) +  1e-7)).mean()
#     #         adv_loss.append(al)

#     else:
#         raise NotImplementedError("v2 dbat is not implemented yet")

#     if reduction == "none":
#         agg_func = func_for_dim(torch.mean, 0)
#     else:
#         assert reduction == "mean"
#         agg_func = torch.mean
#     return aggregate_tensors_by_func(adv_loss, func=agg_func)
