import torch
# import sys
# import os
# import timm
# import torchvision
# import torch.nn.functional as F
import copy
from stuned.utility.utils import (
    get_with_assert,
    aggregate_tensors_by_func,
    func_for_dim,
    # extract_list_from_huge_string
    # append_dict
)


# # local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# from utility.utils import (
#     ChildrenForPicklingPreparer,
#     raise_unknown
# )
# from local_models.baselines import (
#     is_mlp
# )
# from utility.helpers_for_tests import (
#     make_dummy_object
# )
# from local_models.beyond_in import (
#     vit_models
# )
# sys.path.pop(0)


INNER_OBJECT_KEY = "ModuleDelegatingWrapper_inner_object"
CUSTOM_ATTRS_KEY = "ModuleDelegatingWrapper_custom_attrs"
# HYPERPARAM_PREFIX = "__hyperparam__"


# class TrainEvalSwitchModel(torch.nn.Module, ChildrenForPicklingPreparer):

#     def __init__(
#         self,
#         train_model,
#         eval_model,
#         train_kwargs,
#         eval_kwargs
#     ):
#         super().__init__()
#         self.train_model = train_model
#         self.eval_model = eval_model
#         self.current_model = self.eval_model
#         self.train_kwargs = train_kwargs
#         self.eval_kwargs = eval_kwargs
#         self.current_kwargs = self.eval_kwargs

#     def train(self):
#         self.current_model = self.train_model
#         self.current_kwargs = self.train_kwargs
#         self.current_model.train()

#     def eval(self):
#         self.current_model = self.eval_model
#         self.current_kwargs = self.eval_kwargs
#         self.current_model.eval()

#     def forward(self, x):
#         return self.current_model(x, **self.current_kwargs)


# def make_train_eval_switch_model(
#     train_model,
#     eval_model,
#     train_kwargs={},
#     eval_kwargs={}
# ):
#     return TrainEvalSwitchModel(
#         train_model,
#         eval_model,
#         train_kwargs,
#         eval_kwargs
#     )


# def separate_classifier_and_featurizer(
#     original_model,
#     block_id=None,
#     copy_model=True
# ):

#     def model_modified_exception():
#         raise Exception("Model modified by separate_classifier_and_featurizer")

#     def is_timm_vit(model):
#         return isinstance(
#             model,
#             (
#                 timm.models.vision_transformer.VisionTransformer,
#                 vit_models
#             )
#         )

#     if copy_model:
#         model = copy.deepcopy(original_model)
#     else:
#         model = original_model

#     if block_id is not None:
#         assert (
#                 is_timm_vit(model)
#             or
#                 isinstance(model, torchvision.models.resnet.ResNet)
#         )

#     if is_mlp(model):
#         featurizer = model
#         classifier = make_dummy_object()
#         classifier.in_features = featurizer.out_features
#     elif is_timm_vit(model):
#         if block_id is not None:
#             classifier = TimmVitFinalLayers(model, block_id)
#             featurizer = TimmVitPartialBlocks(model, block_id)
#             # model.forward = lambda x: model_modified_exception()
#         else:
#             classifier = model.head
#             featurizer = TimmVitFeaturesWrapper(model)
#             model.forward = lambda x: model_modified_exception()
#     elif isinstance(model, torchvision.models.resnet.ResNet):
#         if block_id is not None:
#             classifier = ResNetPartialBlocks(model, block_id, is_classifier=True)
#             featurizer = ResNetPartialBlocks(model, block_id, is_classifier=False)
#         else:
#             assert copy_model
#             classifier = model.fc
#             featurizer = model
#             featurizer.fc = torch.nn.Identity()
#     else:
#         raise_unknown(
#             "model type",
#             type(model),
#             "separate_classifier_and_featurizer"
#         )
#     return featurizer, classifier


# def assert_block_idx(block_idx, model):
#     assert block_idx == -1  # TODO(Alex | 15.02.2024): relax requirement?
#     # assert block_idx < 0
#     # assert abs(block_idx) < len(model.blocks)


# class ResNetPartialBlocks(torch.nn.Module):

#     def __init__(self, model, block_id, is_classifier=True):
#         super().__init__()
#         blocks = torch.nn.ModuleList()
#         for i, layer in enumerate(model.children()):
#             blocks.append(layer)

#         self.is_classifier = is_classifier

#         if self.is_classifier:
#             self.blocks = blocks[block_id:]
#         else:
#             self.blocks = blocks[:block_id]

#     def forward(self, x):

#         for i, block in enumerate(self.blocks):

#             if self.is_classifier and i + 1 == len(self.blocks):
#                 x = x.squeeze(-1).squeeze(-1)
#             x = block(x)

#         return x


# class TimmVitPartialBlocks(torch.nn.Module):

#     def __init__(self, model, block_idx):
#         super().__init__()
#         assert_block_idx(block_idx, model)
#         self.blocks = model.blocks[:block_idx]
#         self.patch_embed = model.patch_embed
#         self.cls_token = model.cls_token
#         self.pos_embed = model.pos_embed

#         last_block = model.blocks[block_idx]
#         self.attn = last_block.attn
#         self.gamma_1 = last_block.gamma_1
#         self.norm1 = last_block.norm1
#         self.drop_path = last_block.drop_path


#     def forward(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)

#         cls_tokens = self.cls_token.expand(B, -1, -1)

#         x = x + self.pos_embed

#         x = torch.cat((cls_tokens, x), dim=1)

#         for i, blk in enumerate(self.blocks):
#             x = blk(x)

#         x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
#         return x[:, 0]  # only cls token to save space by 197 times


# class TimmVitFinalLayers(torch.nn.Module):

#     def __init__(self, model, block_idx):
#         super().__init__()
#         assert_block_idx(block_idx, model)

#         last_block = model.blocks[block_idx]
#         self.drop_path = last_block.drop_path
#         self.gamma_2 = last_block.gamma_2
#         self.mlp = last_block.mlp
#         self.norm2 = last_block.norm2

#         self.norm = model.norm
#         self.dropout_rate = model.dropout_rate
#         self.head = model.head

#     def forward(self, x, pre_logits=False):

#         x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

#         x = self.norm(x)
#         # x = x[:, 0]
#         if self.dropout_rate:
#             x = F.dropout(x,
#                           p=float(self.dropout_rate),
#                           training=self.training)
#         return x if pre_logits else self.head(x)


# class TimmVitFeaturesWrapper(torch.nn.Module):

#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         # unrequire_grads(self.model, ["head.weight", "head.bias"])
#         self.model.head = None

#     def forward(self, x):
#         x = self.model.forward_features(x)
#         x = self.model.forward_head(x, pre_logits=True)
#         return x


def unrequire_grads(model, parameter_names):
    if parameter_names is not None:
        parameter_names = set(parameter_names)
    for name, param in model.named_parameters():
        if parameter_names is None or name in parameter_names:
            param.requires_grad = False


# TODO(Alex | 17.01.2024): make this non-experimental
# by directly accessing methods like "to" and "eval" from inner_object
# maybe don't even need to inherit from torch.nn.Module,
# just return isinstance of inner_object?
class ModuleDelegatingWrapper(torch.nn.Module):

    def __init__(self, inner_object):
        super().__init__()

        object.__setattr__(self, CUSTOM_ATTRS_KEY, {})
        attrs = self.get_custom_attrs()
        attrs[INNER_OBJECT_KEY] = inner_object

    def __getattr__(self, name):

        inner_object = self.get_inner_object()
        object.__getattribute__(inner_object, name)

    def get_inner_object(self):
        attrs = self.get_custom_attrs()
        return attrs[INNER_OBJECT_KEY]

    def get_custom_attrs(self):
        return object.__getattribute__(self, CUSTOM_ATTRS_KEY)

    def __setattr__(self, key, value):

        inner_object = self.get_inner_object()
        setattr(inner_object, key, value)

    def to(self, *args):
        inner_object = self.get_inner_object()
        inner_object.to(*args)

    def train(self, *args):
        inner_object = self.get_inner_object()
        inner_object.train(*args)

    def eval(self, *args):
        inner_object = self.get_inner_object()
        inner_object.eval(*args)


# TODO(Alex | 24.07.2024): make this non-experimental
class ModelClassesWrapper(ModuleDelegatingWrapper):

    def __init__(self, model, make_mapper):
        super().__init__(model)
        attrs = self.get_custom_attrs()
        attrs["softmax"] = torch.nn.Softmax(dim=-1)
        attrs["mapper"] = make_mapper()

    def __call__(self, x):
        attrs = self.get_custom_attrs()
        underlying_model = self.get_inner_object()
        softmax = attrs["softmax"]
        mapper = attrs["mapper"]

        logits = underlying_model(x)
        probs = softmax(logits)
        return mapper(probs)


def make_model_classes_wrapper(model, make_mapper):
    return ModelClassesWrapper(model, make_mapper)


# based on https://github.com/bethgelab/model-vs-human/blob/master/modelvshuman/datasets/decision_mappings.py
class ToClassesMapping:

    def __init__(self, indices_for_category, aggregation_function=torch.mean):

        self.aggregation_function = aggregation_function
        self.indices_for_category = indices_for_category

    def check_input(self, probabilities):
        assert (probabilities >= 0.0).all() and (probabilities <= 1.0).all()

    def __call__(self, probabilities):

        aggregated_class_probabilities = []

        for category in self.categories:
            indices = self.indices_for_category(category)
            values = probabilities[:, indices]
            aggregated_value = self.aggregation_function(values, axis=-1)
            aggregated_class_probabilities.append(aggregated_value.unsqueeze(1))

        aggregated_class_probabilities = torch.cat(
            aggregated_class_probabilities,
            dim=1
        )

        return aggregated_class_probabilities


def make_to_classes_mapping(indices_for_category, aggregation_function=torch.mean):
    return ToClassesMapping(indices_for_category, aggregation_function)


class ModelBuilderBase:
    def build(self):
        raise NotImplementedError()


class ModelBuilderFromList(ModelBuilderBase):

    def __init__(self, list_of_models):
        self.list_of_models = list_of_models
        self.total_models = len(self.list_of_models)
        self.num_returned_models = 0
        for model in list_of_models:
            assert isinstance(model, torch.nn.Module)

    def build(self):
        assert self.num_returned_models < self.total_models, \
            "Already returned all models from the list."
        model = self.list_of_models[self.num_returned_models]
        self.num_returned_models += 1
        return model


def get_model(path, base_model=None, patch_model=None):
    model = torch.load(path)
    if isinstance(model, dict):
        if "model" in model:
            model = get_with_assert(model, "model")
        else:
            assert base_model is not None
            if "state_dict" in model:
                model = model["state_dict"]
            base_model = copy.deepcopy(base_model)
            base_model.load_state_dict(model)
            model = base_model
    if patch_model is not None:
        model = patch_model(model)
    model.to("cpu")
    return model


def make_model_builder_from_list(list_of_models):
    return ModelBuilderFromList(list_of_models)


# def make_models_dict_from_huge_string(huge_string, keys, id_prefix=""):
#     # TODO(Alex | 02.04.2024): maybe later we can extend it to the whole gsheet
#     # and update it by adding new columns

#     def make_id(id_prefix, keys, split):
#         id_name = id_prefix
#         assert "path" in keys
#         path = None
#         properties = {}
#         for i, (key, value) in enumerate(zip(keys, split)):
#             if key != "path":
#                 id_name += key + '_' + value
#                 if i + 1 != len(keys):
#                     id_name += '_'
#                 properties[HYPERPARAM_PREFIX + key] = value
#             else:
#                 path = value
#         return id_name, path, properties

#     res = {}
#     name_to_prop = {}

#     huge_string = huge_string.replace('\t', ' ').replace('\n', ' ')

#     split = huge_string.split()

#     if keys is None:
#         assert id_prefix != ""
#         return {id_prefix: split}, {id_prefix: {}}

#     assert len(split) % len(keys) == 0, \
#             f"split {split} is not suitable for keys {keys}"

#     current_tuple = []
#     for item in split:

#         current_tuple.append(item)
#         if len(current_tuple) < len(keys):
#             continue

#         id_name, path, properties = make_id(id_prefix, keys, current_tuple)

#         if id_name in name_to_prop:
#             assert name_to_prop[id_name] == properties
#         else:
#             name_to_prop[id_name] = properties

#         assert path is not None

#         append_dict(res, {id_name: path}, allow_new_keys=True)

#         current_tuple = []

#     return res, name_to_prop


def stores_input(outputs):
    assert len(outputs) > 0
    output_0 = outputs[0]
    return isinstance(output_0, (list, tuple)) and len(output_0) == 2


def compute_ensemble_output(
    outputs,
    weights=None,
    process_logits=None
):

    if process_logits is None:
        process_logits = lambda x: x

    if weights is None:
        weights = [1.0] * len(outputs)

    if stores_input(outputs):
        extractor = lambda x: x[1]
    else:
        extractor = lambda x: x

    return aggregate_tensors_by_func(
        [
            weight * process_logits(extractor(submodel_output).unsqueeze(0))
                for weight, submodel_output
                    in zip(weights, outputs)
        ],
        func=func_for_dim(torch.mean, dim=0)
    ).squeeze(0)


def bootstrap_ensemble_outputs(outputs, assert_len=True):
    if_stores_input = stores_input(outputs)
    if assert_len:
        assert if_stores_input
    if if_stores_input:
        return [output[1] for output in outputs]
    else:
        return outputs
