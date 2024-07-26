import torch
import sys
# import os
import copy
import random
from stuned.utility.utils import get_project_root_path
import torch.nn as nn
from stuned.utility.utils import (
    aggregate_tensors_by_func,
    func_for_dim,
    extract_list_from_huge_string
#     apply_random_seed
)


# local modules
sys.path.insert(0, get_project_root_path())
from diverse_universe.local_models.utils import ModelBuilderBase
from diverse_universe.local_models.baselines import (
    is_mlp
)
from diverse_universe.local_models.utils import (
    ModelClassesWrapper,
    unrequire_grads,
    get_model,
    make_model_builder_from_list,
    compute_ensemble_output
)
# from diverse_universe.local_datasets.utils import (
#     get_probs
# )
from diverse_universe.train.losses import (
    get_probs
)
# from train_eval.utils import (
#     compute_ensemble_output
# )
sys.path.pop(0)


REDNECK_ENSEMBLE_KEY = "redneck_ensemble"
SINGLE_MODEL_KEY = "single_model_per_epoch"
POE_KEY = "product_of_experts"


def cycle_id(current_id, total_len):
    new_id = current_id + 1
    if new_id == total_len:
        new_id = 0
    return new_id


# based on: https://github.com/mlfoundations/model-soups/blob/main/main.py#L114-L125
def make_uniform_soup(models):
    num_models = len(models)
    for j, submodel in enumerate(models):

        state_dict = submodel.state_dict()
        num_models = len(models)
        if j == 0:
            uniform_soup = {
                k : v.cpu() * (1./num_models) for k, v in state_dict.items()
            }
        else:
            uniform_soup = {
                k : v.cpu() * (1./num_models) + uniform_soup[k]
                    for k, v in state_dict.items()
            }
    return uniform_soup


def split_linear_layer(linear_layer, n_heads):
    list_of_sublayers = []
    in_features = linear_layer.in_features
    out_features = linear_layer.out_features

    assert out_features % n_heads == 0

    out_features_per_head = out_features // n_heads
    for i in range(n_heads):
        sublayer = torch.nn.Linear(
            in_features,
            out_features_per_head
        )

        current_slice = (i * out_features_per_head, (i + 1) * out_features_per_head)
        sublayer.weight = torch.nn.Parameter(
            linear_layer.weight[current_slice[0]: current_slice[1], :]
        )
        sublayer.bias = torch.nn.Parameter(
            linear_layer.bias[current_slice[0]: current_slice[1]]
        )
        list_of_sublayers.append(sublayer)
    return list_of_sublayers


class RedneckEnsemble(nn.Module):
    def __init__(
        self,
        n_models,
        base_model_builder,
        weights=None,
        single_model_per_epoch=False,
        identical=False,
        feature_extractor=None,
        product_of_experts=False,
        random_select=None,
        keep_inactive_on_cpu=False,
        softmax_ensemble=False,
        split_last_linear_layer=False,
        freeze_feature_extractor=True
    ):
        def one_layer_model(submodels):
            for submodel in submodels:
                if not is_mlp(submodel):
                    return False
                # expect only linear + dropout
                if len(submodel.mlp._modules) > 2:
                    return False
            return True

        assert n_models > 0, "Need to have at least one model in ensemble"
        assert not (single_model_per_epoch and product_of_experts), \
            f"Cannot use {SINGLE_MODEL_KEY} and {POE_KEY} together"
        if single_model_per_epoch or product_of_experts:
            assert weights is None, \
                f"Cannot use weights with {SINGLE_MODEL_KEY} or {POE_KEY}"

        if single_model_per_epoch:
            self.single_model_id = 0
        else:
            self.single_model_id = None
        assert isinstance(base_model_builder, ModelBuilderBase)

        super(RedneckEnsemble, self).__init__()
        self.n_models = n_models
        self.split_last_linear_layer = split_last_linear_layer
        if identical:
            models_list = [base_model_builder.build()]
            for _ in range(self.n_models - 1):
                models_list.append(copy.deepcopy(models_list[0]))
        else:
            if self.split_last_linear_layer:
                linear_layer = base_model_builder.build()
                assert isinstance(linear_layer, nn.Linear)
                models_list = split_linear_layer(linear_layer, self.n_models)
            else:
                models_list = [
                    base_model_builder.build() for _ in range(self.n_models)
                ]
        self.submodels = nn.ModuleList(
            models_list
        )
        self.set_weights(weights)

        self.freeze_feature_extractor = freeze_feature_extractor

        self.feature_extractor = feature_extractor
        if self.feature_extractor is not None and self.freeze_feature_extractor:
            unrequire_grads(self.feature_extractor, None)

        self.product_of_experts = product_of_experts
        self.softmax = torch.nn.Softmax(dim=-1)

        self.random_select = random_select
        if self.random_select is not None:
            assert self.random_select > 1 and self.random_select <= self.n_models

        self.one_layer_models = one_layer_model(self.submodels)
        self.latest_device = torch.device("cpu")
        self.keep_inactive_on_cpu = keep_inactive_on_cpu
        self.different_devices = False

        self.active_indices = None
        self.keep_active_indices = False

        self.soup = None
        self.softmax_ensemble = softmax_ensemble

    def apply_feature_extractor(self, x):
        if (
                hasattr(self, "feature_extractor")
            and
                self.feature_extractor is not None
        ):
            x = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        return x

    def cook_soup(self):
        assert len(self.submodels) > 0
        base_model = self.submodels[0]
        if isinstance(base_model, ModelClassesWrapper):
            base_model = base_model.get_inner_object()
        self.soup = copy.deepcopy(base_model)
        self.soup.load_state_dict(make_uniform_soup(self.submodels))

    def set_weights(self, weights, normalize=True):
        self.weights = weights

        if self.weights is not None and normalize:
            self.weights = normalize_weights(torch.Tensor(self.weights))

    def set_single_model_id(self, id):
        assert id >= 0 and id < self.n_models, \
            f"Invalid model id {id}, have only {self.n_models} models"
        self.single_model_id = id

    def after_epoch(self, is_train, logger):
        if is_train and self.single_model_id is not None:
            new_id = cycle_id(self.single_model_id, self.n_models)
            logger.log(f"Switching single model id to {new_id}")
            self.set_single_model_id(new_id)

    def to(self, *args, **kwargs):
        if isinstance(args[0], torch.device):
            self.latest_device = args[0]

        # TODO(Alex | 24.01.2024): make sure that
        # torch.nn.module recursively calls this automatically
        # once ModuleDelegatingWrapper is fixed
        # in case submodels are wrapped
        for submodel in self.submodels:
            submodel.to(*args, **kwargs)
        if hasattr(self, "soup") and self.soup is not None:
            self.soup.to(*args, **kwargs)
        super().to(*args, **kwargs)

    def forward(self, input):

        def equivalent_linear(active_submodels):
            cat_weight = []
            cat_bias = []
            for submodel in active_submodels:
                assert len(submodel.mlp._modules) == 2, \
                    "Expect only linear + dropout"
                linear_layer = submodel.mlp._modules["0"]
                cat_weight.append(linear_layer.weight)
                cat_bias.append(linear_layer.bias)

            cat_weight = torch.cat(cat_weight, dim=0)
            cat_bias = torch.cat(cat_bias, dim=0)
            cat_output = torch.nn.functional.linear(input, cat_weight, cat_bias)
            outputs = torch.chunk(cat_output, len(active_submodels), dim=-1)
            outputs = [[input, submodel_output] for submodel_output in outputs]
            return outputs

        def apply_submodel(submodel, input):
            if isinstance(submodel, torch.nn.Linear):
                input = input.squeeze(-1).squeeze(-1)
            return submodel(input)

        def select_active_submodels(
            submodels,
            random_select,
            training,
            keep_inactive_on_cpu,
            latest_device,
            different_devices,
            keep_active_indices,
            active_indices
        ):
            if not keep_active_indices:
                active_indices = None
            if training and random_select is not None:
                if active_indices is None:
                    active_indices = set(random.sample(
                        range(len(submodels)),
                        random_select
                    ))

                active_submodels = []
                for i in range(len(submodels)):
                    if i in active_indices:
                        active_submodels.append(submodels[i].to(latest_device))
                    elif keep_inactive_on_cpu:
                        different_devices = True
                        submodels[i].to("cpu")
            else:
                if different_devices:
                    for i in range(len(submodels)):
                        submodels[i].to(latest_device)
                    different_devices = False
                active_submodels = submodels
            return active_submodels, different_devices, active_indices

        if self.feature_extractor is not None:
            input = self.apply_feature_extractor(input)

        if self.single_model_id is not None and self.training:
            assert self.product_of_experts is False
            return self.submodels[self.single_model_id](input)

        if self.product_of_experts and self.training:
            assert hasattr(self, "softmax")
            return aggregate_tensors_by_func(
                [
                    torch.log(self.softmax(submodel(input)))
                        for submodel in self.submodels
                ],
                func=func_for_dim(torch.sum, dim=0)
            )

        (
            active_submodels,
            self.different_devices,
            self.active_indices
        ) = select_active_submodels(
            self.submodels,
            self.random_select,
            self.training,
            self.keep_inactive_on_cpu,
            self.latest_device,
            self.different_devices,
            self.keep_active_indices,
            self.active_indices
        )

        if self.training and self.one_layer_models:
            outputs = equivalent_linear(active_submodels)
        else:
            outputs = [
                [input, apply_submodel(submodel, input)]
                    for submodel
                        in active_submodels
            ]
        if self.weights is not None:
            if self.softmax_ensemble:
                process_logits = get_probs
            else:
                process_logits = None
            assert len(self.weights) == len(outputs)

            aggregated_output = compute_ensemble_output(
                outputs,
                self.weights,
                process_logits=process_logits
            )
            outputs.append([input, aggregated_output])
        if self.n_models == 1:
            assert len(outputs) == 1, \
                "n_models is 1, but number of outputs is not equal to 1, " \
                "maybe ensemble mode is on"

            assert len(outputs[0]) == 2
            outputs = outputs[0][1]
        return outputs


def make_redneck_ensemble(
    n_models,
    base_model_builder,
    weights=None,
    single_model_per_epoch=False,
    identical=False,
    feature_extractor=None,
    product_of_experts=False,
    random_select=None,
    keep_inactive_on_cpu=False,
    split_last_linear_layer=False,
    freeze_feature_extractor=True
):
    return RedneckEnsemble(
        n_models,
        base_model_builder,
        weights,
        single_model_per_epoch,
        identical,
        feature_extractor,
        product_of_experts,
        random_select=random_select,
        keep_inactive_on_cpu=keep_inactive_on_cpu,
        split_last_linear_layer=split_last_linear_layer,
        freeze_feature_extractor=freeze_feature_extractor
    )


def normalize_weights(weights, p=2):
    assert len(weights.shape) == 1
    if not torch.isclose(torch.linalg.norm(weights, p), torch.Tensor([1])).all():
        weights = torch.nn.functional.normalize(weights, p=p, dim=0)
    return weights


def is_ensemble(model):
    return isinstance(model, RedneckEnsemble)


# def stores_input(outputs):
#     assert len(outputs) > 0
#     output_0 = outputs[0]
#     return isinstance(output_0, (list, tuple)) and len(output_0) == 2


# def compute_ensemble_output(
#     outputs,
#     weights=None,
#     process_logits=None
# ):

#     if process_logits is None:
#         process_logits = lambda x: x

#     if weights is None:
#         weights = [1.0] * len(outputs)

#     if stores_input(outputs):
#         extractor = lambda x: x[1]
#     else:
#         extractor = lambda x: x

#     return aggregate_tensors_by_func(
#         [
#             weight * process_logits(extractor(submodel_output).unsqueeze(0))
#                 for weight, submodel_output
#                     in zip(weights, outputs)
#         ],
#         func=func_for_dim(torch.mean, dim=0)
#     ).squeeze(0)


def make_ensembles_from_paths(paths, group_by, num_ensembles, base_model=None):
    paths = extract_list_from_huge_string(paths)
    total_paths = len(paths)
    assert total_paths >= num_ensembles
    assert total_paths >= group_by

    indices_per_ensemble = []

    # select indices
    for j in range(num_ensembles):
        indices_per_ensemble.append(
            set(random.sample(list(range(total_paths)), group_by))
        )

    res = [[] for _ in range(num_ensembles)]

    for i, path in enumerate(paths):
        for j in range(len(indices_per_ensemble)):
            if i in indices_per_ensemble[j]:
                res[j].append(get_ensemble(path, base_model=base_model))

    return [make_ensemble_from_model_list(model_list) for model_list in res]


def make_ensemble_from_model_list(list_of_models):
    return make_redneck_ensemble(
        len(list_of_models),
        make_model_builder_from_list(list_of_models)
    )


def get_ensemble(path, base_model=None):
    return get_model(path, base_model=base_model, patch_model=patch_ensemble)


def patch_ensemble(model):
    if is_ensemble(model):
        model.keep_inactive_on_cpu = False
        model.latest_device = torch.device("cpu")
        model.different_devices = False
        model.random_select = None
        model.keep_active_indices = False
        model.active_indices = None
        model.softmax = torch.nn.Softmax(dim=-1)
        model.softmax_ensemble = False
    return model


def make_ensembles(paths):
    if isinstance(paths, str):
        paths = extract_list_from_huge_string(paths)
    res = []
    for path in paths:
        res.append(get_model(path))
    return res
