import os
import sys
import torch
import torchvision
from torchvision import models
from torchvision.models.resnet import (
    ResNet,
    ResNet50_Weights,
    ResNet18_Weights
)
import timm
import copy
import torch.nn.functional as F
from stuned.utility.utils import (
    get_project_root_path,
    get_with_assert,
    append_dict,
    raise_unknown,
    # extract_list_from_huge_string,
    # read_json
)
from stuned.utility.logger import (
    make_logger
)
from stuned.utility.helpers_for_tests import (
    make_dummy_object
)


# local modules
sys.path.insert(0, get_project_root_path())
from diverse_universe.local_models.ensemble import (
    REDNECK_ENSEMBLE_KEY,
    SINGLE_MODEL_KEY,
    POE_KEY,
    make_redneck_ensemble,
    is_ensemble,
    # split_linear_layer
)
from diverse_universe.local_models.wrappers import (
    # wrap_model,
    make_mvh_wrapper,
    make_in9_wrapper,
    make_ina_wrapper,
    make_inr_wrapper
)
# from utility.logger import (
#     make_logger
# )
# from utility.utils import (
    # raise_unknown,
    # get_with_assert,
#     get_project_root_path,
    # read_json
# )
# from utility.utils import (
#     extract_list_from_huge_string
# )
# from utility.configs import (
#     ANY_KEY
# )
# from local_models.base import ModelBuilderBase
# from local_models.diverse_vit import (
#     make_diverse_vit,
#     make_masking_wrapper
# )
from diverse_universe.local_models.utils import (
    ModelBuilderBase,
    get_model
#     TrainEvalSwitchModel,
#     make_train_eval_switch_model,
#     separate_classifier_and_featurizer,
    # make_model_classes_wrapper,
    # make_to_classes_mapping,
)
# from diverse_universe.local_models.wrappers import (
#     make_mvh_model_wrapper
# )
from diverse_universe.local_models.baselines import (
    is_mlp,
    make_mlp
)
from diverse_universe.local_models.deit3 import (
    vit_models,
    load_model_transform,
)
# from local_datasets.utils import (
#     JSON_PATH
# )
# from local_models.model_vs_human import make_mvh_model_wrapper
sys.path.pop(0)


RESNET_DEFAULT_N_CHANNELS = 3
RESNET_DEFAULT_N_CLASSES = 1000
# IN9_MAPPING_PATH = os.path.join(
#     get_project_root_path(),
#     "json",
#     "in_to_in9.json"
# )
# IN9_CATEGOREIS = tuple([
#     "dog",
#     "bird",
#     "wheeled vehicle",
#     "reptile",
#     "carnivore",
#     "insect",
#     "musical instrument",
#     "primate",
#     "fish"
# ])
# IMAGENET_A_JSON = os.path.join(JSON_PATH, "imagenet_a_wnids.json")
# IMAGENET_R_JSON = os.path.join(JSON_PATH, "imagenet_r_wnids.json")
HYPERPARAM_PREFIX = "__hyperparam__"


class ModelBuilder(ModelBuilderBase):
    def __init__(self, model_config, logger=None):

        if logger is None:
            logger = make_logger()

        self.model_type = model_config["type"]
        assert self.model_type in model_config
        self.model_specific_config = model_config[self.model_type]

        self.separation_config = model_config.get("separation_config", {})

        self.logger = logger

        if "resnet" in self.model_type:
            self.pretrained = get_with_assert(
                self.model_specific_config,
                "pretrained"
            )
            self.n_channels = get_with_assert(
                self.model_specific_config,
                "n_channels"
            )
            self.n_classes = get_with_assert(
                self.model_specific_config,
                "n_classes"
            )
            if self.pretrained:
                assert self.n_channels == RESNET_DEFAULT_N_CHANNELS
            if self.model_type == "resnet50":
                self.build_func = self._build_resnet50
            elif self.model_type == "resnet18":
                self.build_func = self._build_resnet18
            else:
                raise_unknown(
                    "resnet model type",
                    self.model_type,
                    "model config"
                )

        elif self.model_type == "linear":
            self.build_func = self._build_linear

        elif self.model_type == REDNECK_ENSEMBLE_KEY:
            self.build_func = self._build_redneck_ensemble

        elif self.model_type == "diverse_vit":
            self.build_func = self._build_diverse_vit

        elif self.model_type == "diverse_vit_switch_ensemble":
            self.head_indices_list = get_with_assert(
                self.model_specific_config,
                "head_indices_list"
            )
            diverse_vit_config = get_with_assert(
                self.model_specific_config,
                "inner_vit"
            )
            self.diverse_vit_builder = ModelBuilder(
                diverse_vit_config,
                logger=self.logger
            )
            self.build_func = self._build_diverse_vit_switch_ensemble

        elif self.model_type == "mlp":
            self.build_func = self._build_mlp

        elif self.model_type == "timm_model":
            self.build_func = self._build_timm_model

        elif self.model_type == "torch_load":
            self.build_func = self._torch_load

        else:
            raise_unknown("model type", self.model_type, "model config")

    def _build_timm_model(self):
        timm_id = get_with_assert(self.model_specific_config, "timm_id")
        timm_kwargs = self.model_specific_config.get(
            "timm_kwargs",
            {}
        )
        return timm.create_model(timm_id, **timm_kwargs)

    def _torch_load(self):
        model_name = get_with_assert(self.model_specific_config, "model_name")
        # TODO(Alex | 22.01.2024) allow different model types
        checkpoint_path = get_with_assert(self.model_specific_config, "path")
        if model_name == "deit3_21k":
            # TODO(Alex | 16.02.2024): move to separate builder for deit3
            # as it does not make sense to have None checkpoint in torch_load
            if checkpoint_path is None:
                pretrained_dir = None
            else:
                pretrained_dir = os.path.dirname(checkpoint_path)
            model, _ = load_model_transform(
                model_name,
                pretrained_dir
            )
        else:
            assert model_name == "resnet18_1000"
            base_model = torchvision.models.resnet18(num_classes=1000)
            if checkpoint_path is None:
                model = base_model
            else:
                model = get_model(
                    checkpoint_path,
                    base_model=base_model
                )

        return model

    # def _build_diverse_vit(self):
    #     return make_diverse_vit(self.model_specific_config)

    def _build_mlp(self):
        return make_mlp(**self.model_specific_config)

    # def _build_diverse_vit_switch_ensemble(self):
    #     assert hasattr(self, "diverse_vit_builder")
    #     assert hasattr(self, "head_indices_list")

    #     diverse_vit = self.diverse_vit_builder.build()

    #     if self.head_indices_list:
    #         for head_indices in self.head_indices_list:
    #             assert any(head_indices), \
    #                 f"At least one head should be used, " \
    #                 f"while given head indices are all False: {head_indices}"
    #     else:
    #         # Extract the number of heads from the diverse_vit instance
    #         num_heads = diverse_vit.vit.last_attn_num_heads
    #         self.head_indices_list = []
    #         # Use all heads
    #         self.head_indices_list.append([True] * num_heads)
    #         # Use one head at a time
    #         for head_idx  in range(num_heads):
    #             head_indices = [False] * num_heads
    #             head_indices[head_idx] = True
    #             self.head_indices_list.append(head_indices)

    #     return make_diverse_vit_switch_ensemble(
    #         diverse_vit,
    #         self.head_indices_list
    #     )

    def _build_resnet18(self):
        return self._build_resnet_n(
            constructor=models.resnet18,
            default_weights=ResNet18_Weights.DEFAULT
        )

    def _build_resnet50(self):
        return self._build_resnet_n(
            constructor=models.resnet50,
            default_weights=ResNet50_Weights.DEFAULT
        )

    def _build_linear(self):
        in_features = get_with_assert(self.model_specific_config, "in_features")
        out_features = get_with_assert(self.model_specific_config, "out_features")
        return torch.nn.Linear(in_features, out_features)

    def _build_resnet_n(self, constructor, default_weights):

        def update_resnet_top_layer(model, n_classes):
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, n_classes)
            return model

        def update_resnet_first_layer_channels(model, n_channels):

            model.conv1 = torch.nn.Conv2d(
                in_channels=n_channels,
                out_channels=model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=model.conv1.bias
            )

            return model

        assert self.pretrained is not None
        assert self.n_classes is not None
        assert self.n_channels is not None

        model = constructor(
            weights=default_weights if self.pretrained else None
        )

        if self.n_channels != RESNET_DEFAULT_N_CHANNELS:
            model = update_resnet_first_layer_channels(
                model=model,
                n_channels=self.n_channels
            )

        if self.n_classes != RESNET_DEFAULT_N_CLASSES:
            model = update_resnet_top_layer(
                model=model,
                n_classes=self.n_classes
            )
        return model

    def _build_redneck_ensemble(self):

        feature_extractor = build_model(
            self.model_specific_config.get("feature_extractor")
        )

        base_estimator_builder = ModelBuilder(
            get_with_assert(
                self.model_specific_config,
                "base_estimator"
            ),
            logger=self.logger
        )
        return make_redneck_ensemble(
            get_with_assert(
                self.model_specific_config,
                "n_models"
            ),
            base_estimator_builder,
            weights=self.model_specific_config.get("weights", None),
            single_model_per_epoch=self.model_specific_config.get(
                SINGLE_MODEL_KEY,
                False
            ),
            identical=self.model_specific_config.get("identical", False),
            feature_extractor=feature_extractor,
            product_of_experts=self.model_specific_config.get(
                POE_KEY,
                False
            ),
            random_select=self.model_specific_config.get(
                "random_select"
            ),
            keep_inactive_on_cpu=self.model_specific_config.get(
                "keep_inactive_on_cpu",
                False
            ),
            split_last_linear_layer=self.model_specific_config.get(
                "split_last_linear_layer",
                False
            ),
            freeze_feature_extractor=self.model_specific_config.get(
                "freeze_feature_extractor",
                True
            )
        )

    def build(self):
        self.logger.log("Building {} model..".format(self.model_type))
        model = self.build_func()
        if len(self.separation_config) != 0:
            self.logger.log(
                f"Separating model with config {self.separation_config}"
            )
            separate_what = self.separation_config.get(
                "separate",
                "classifier"
            )

            feature_extractor, classifier = separate_classifier_and_featurizer(
                model,
                block_id=self.separation_config.get("block_id")
            )
            if separate_what == "classifier":
                model = classifier
            else:
                assert separate_what == "feature_extractor"
                model = feature_extractor
        return model


# def make_diverse_vit_switch_ensemble(
#     diverse_vit,
#     head_indices_list
# ):
#     diverse_vit_as_ensemble = make_ensemble_from_model_list(
#         [
#             make_masking_wrapper(diverse_vit, head_indices)
#                 for head_indices
#                 in head_indices_list
#         ]
#     )

#     return make_train_eval_switch_model(
#         diverse_vit,
#         diverse_vit_as_ensemble,
#         train_kwargs={"return_io": True}
#     )


# def resize_input(
#     input,
#     target_shape,
#     interpolation=torchvision.transforms.InterpolationMode.BILINEAR
# ):
#     assert len(input.shape) == 4
#     assert len(target_shape) == 3 or len(target_shape) == 2

#     if len(target_shape) == 3:
#         target_shape = target_shape[1:]

#     if input.shape[2:] != target_shape:
#         input = torchvision.transforms.Resize(
#             target_shape,
#             interpolation=interpolation,
#             max_size=None,
#             antialias='warn'
#         )(input)
#     return input


# class ModelWrapper(torch.nn.Module):

#     def __init__(self, model):
#         super().__init__()
#         self.underlying_model = model


# class ResizingModelWrapper(ModelWrapper):

#     def __init__(self, model, target_shape):
#         super().__init__(model)
#         self.target_shape = target_shape

#     def forward(self, x):
#         return self.underlying_model(resize_input(x, self.target_shape))


# class TransformsWrapper(ModelWrapper):

#     def __init__(self, model, transforms):
#         super().__init__(model)
#         self.transforms = transforms

#     def forward(self, x):
#         return self.underlying_model(self.transforms(x))


# def make_resizing_model_wrapper(model, target_shape):
#     return ResizingModelWrapper(model, target_shape)


# def make_transforms_wrapper(model, transforms):
#     return TransformsWrapper(model, transforms)


def make_model_builder(model_config, logger):
    return ModelBuilder(model_config=model_config, logger=logger)


def build_model(model_config, logger=None):

    if model_config is None:
        return None

    model_builder = make_model_builder(model_config, logger)
    model = model_builder.build()

    model = wrap_model(model, model_config.get("wrapper"))

    return model


# def get_num_classes(model):
#     if isinstance(model, ResNet):
#         return model.fc.weight.shape[0]
#     elif is_ensemble(model):
#         return get_num_classes(model.submodels[0])
#     else:
#         raise Exception(
#             "Can't infer num_classes from model {}".format(type(model))
#         )


# def check_model(model, expected_model):

#     if expected_model == REDNECK_ENSEMBLE_KEY:
#         assert \
#             is_ensemble(model), \
#             "Expected RedneckEnsemble"
#     elif "resnet" in expected_model:
#         assert \
#             isinstance(model, ResNet), \
#             "Expected ResNet"
#     elif "diverse_vit_switch_ensemble" in expected_model:
#         assert \
#             isinstance(model, TrainEvalSwitchModel), \
#             "Expected TrainEvalSwitchModel"
#     elif "mlp" == expected_model:
#         assert \
#             is_mlp(model), \
#             "Expected MLP"
#     else:
#         assert expected_model == ANY_KEY, \
#             f"Model type \"{expected_model}\" specified in models' config " \
#             f"is not among known models and not \"{ANY_KEY}\""


# class ModelBuilderFromList(ModelBuilderBase):

#     def __init__(self, list_of_models):
#         self.list_of_models = list_of_models
#         self.total_models = len(self.list_of_models)
#         self.num_returned_models = 0
#         for model in list_of_models:
#             assert isinstance(model, torch.nn.Module)

#     def build(self):
#         assert self.num_returned_models < self.total_models, \
#             "Already returned all models from the list."
#         model = self.list_of_models[self.num_returned_models]
#         self.num_returned_models += 1
#         return model


# def make_model_builder_from_list(list_of_models):
#     return ModelBuilderFromList(list_of_models)


# class DummyConstModel(torch.nn.Module):

#     def __init__(self, n):
#         super(DummyConstModel, self).__init__()
#         self.n = n

#     def forward(self, x):
#         return self.n


# def make_ensemble_from_model_list(list_of_models):
#     return make_redneck_ensemble(
#         len(list_of_models),
#         make_model_builder_from_list(list_of_models)
#     )


# def make_ensemble_from_stacked_linear_layers(stacked_model, n_heads):

#     feature_extractor, classifier = separate_classifier_and_featurizer(
#         stacked_model
#     )
#     assert isinstance(classifier, torch.nn.Linear)
#     list_of_classifiers = split_linear_layer(classifier, n_heads)
#     ensemble = make_ensemble_from_model_list(list_of_classifiers)
#     ensemble.feature_extractor = feature_extractor
#     return ensemble


# def wrap_model(model, wrapper_type):

#     if wrapper_type is None:
#         return model

#     if is_ensemble(model):
#         model = copy.deepcopy(model)
#         for i in range(len(model.submodels)):
#             model.submodels[i] = wrap_model(
#                 model.submodels[i],
#                 wrapper_type
#             )
#         if hasattr(model, "soup") and model.soup is not None:
#             model.soup = wrap_model(model.soup, wrapper_type)
#     else:
#         if wrapper_type == "mvh":
#             model = make_mvh_model_wrapper(model)
#         elif wrapper_type == "IN9":
#             model = make_in9_wrapper(model)
#         elif wrapper_type == "ina":
#             model = make_ina_wrapper(model)
#         elif wrapper_type == "inr":
#             model = make_inr_wrapper(model)
#         else:
#             raise_unknown("wrapper type", wrapper_type, "wrapper config")

#     return model


# def invert_dict_with_repetitions(d):
#     res = {}
#     for k, v in d.items():
#         res.setdefault(v, [])
#         res[v].append(k)
#     return res


# class IN9Categories:

#     def __init__(self):
#         map_to_in9 = read_json(IN9_MAPPING_PATH)
#         self.in1000_to_in9 = invert_dict_with_repetitions(map_to_in9)
#         self.in1000_to_in9 = {
#             k: [int(vi) for vi in v] for k, v in self.in1000_to_in9.items()
#         }

#     def __call__(self, category_id):
#         return self.in1000_to_in9[category_id]


# def make_in9_mapper():
#     mapper = make_to_classes_mapping(
#         IN9Categories()
#     )
#     mapper.categories = list(range(len(IN9_CATEGOREIS)))
#     return mapper


# def make_in9_wrapper(model):
#     return make_model_classes_wrapper(model, make_in9_mapper)


# class ImagenetARMapper:

#     def __init__(self, mapper_type, argmax_before=True):

#         if mapper_type == "ina":
#             json = read_json(IMAGENET_A_JSON)
#             key = "a"
#         else:
#             assert mapper_type == "inr"
#             key = "r"
#             json = read_json(IMAGENET_R_JSON)

#         self.imagenet_ar_mask = [
#             wnid in json[f"imagenet_{key}_wnids"]
#             for wnid
#             in json["all_wnids"]
#         ]
#         self.argmax_before = argmax_before

#     def __call__(self, probabilities):

#         if self.argmax_before:
#             argmax = probabilities.argmax(-1)
#             probabilities = torch.zeros_like(probabilities)
#             probabilities[torch.arange(probabilities.shape[0]), argmax] = 1

#         return probabilities[:, self.imagenet_ar_mask]


# def make_imagenet_ar_mapper(mapper_type):

#     def make_mapper():
#         return ImagenetARMapper(mapper_type)

#     return make_mapper


# def make_ina_wrapper(model):
#     return make_model_classes_wrapper(model, make_imagenet_ar_mapper("ina"))


# def make_inr_wrapper(model):
#     return make_model_classes_wrapper(model, make_imagenet_ar_mapper("inr"))


# def patch_model(model):
#     if is_ensemble(model):
#         model.keep_inactive_on_cpu = False
#         model.latest_device = torch.device("cpu")
#         model.different_devices = False
#         model.random_select = None
#         model.keep_active_indices = False
#         model.active_indices = None
#         model.softmax = torch.nn.Softmax(dim=-1)
#         model.softmax_ensemble = False
#     return model


# def get_model(path, base_model=None):
#     model = torch.load(path)
#     if isinstance(model, dict):
#         if "model" in model:
#             model = get_with_assert(model, "model")
#         else:
#             assert base_model is not None
#             if "state_dict" in model:
#                 model = model["state_dict"]
#             base_model = copy.deepcopy(base_model)
#             base_model.load_state_dict(model)
#             model = base_model
#     model = patch_model(model)
#     model.to("cpu")
#     return model


# def make_ensembles(paths):
#     if isinstance(paths, str):
#         paths = extract_list_from_huge_string(paths)
#     res = []
#     for path in paths:
#         res.append(get_model(path))
#     return res


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


def make_models_dict_from_huge_string(huge_string, keys, id_prefix=""):
    # TODO(Alex | 02.04.2024): maybe later we can extend it to the whole gsheet
    # and update it by adding new columns

    def make_id(id_prefix, keys, split):
        id_name = id_prefix
        assert "path" in keys
        path = None
        properties = {}
        for i, (key, value) in enumerate(zip(keys, split)):
            if key != "path":
                id_name += key + '_' + value
                if i + 1 != len(keys):
                    id_name += '_'
                properties[HYPERPARAM_PREFIX + key] = value
            else:
                path = value
        return id_name, path, properties

    res = {}
    name_to_prop = {}

    huge_string = huge_string.replace('\t', ' ').replace('\n', ' ')

    split = huge_string.split()

    if keys is None:
        assert id_prefix != ""
        return {id_prefix: split}, {id_prefix: {}}

    assert len(split) % len(keys) == 0, \
            f"split {split} is not suitable for keys {keys}"

    current_tuple = []
    for item in split:

        current_tuple.append(item)
        if len(current_tuple) < len(keys):
            continue

        id_name, path, properties = make_id(id_prefix, keys, current_tuple)

        if id_name in name_to_prop:
            assert name_to_prop[id_name] == properties
        else:
            name_to_prop[id_name] = properties

        assert path is not None

        append_dict(res, {id_name: path}, allow_new_keys=True)

        current_tuple = []

    return res, name_to_prop


def assert_block_idx(block_idx, model):
    assert block_idx == -1  # TODO(Alex | 15.02.2024): relax requirement?
    # assert block_idx < 0
    # assert abs(block_idx) < len(model.blocks)


class TimmVitFinalLayers(torch.nn.Module):

    def __init__(self, model, block_idx):
        super().__init__()
        assert_block_idx(block_idx, model)

        last_block = model.blocks[block_idx]
        self.drop_path = last_block.drop_path
        self.gamma_2 = last_block.gamma_2
        self.mlp = last_block.mlp
        self.norm2 = last_block.norm2

        self.norm = model.norm
        self.dropout_rate = model.dropout_rate
        self.head = model.head

    def forward(self, x, pre_logits=False):

        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        x = self.norm(x)
        # x = x[:, 0]
        if self.dropout_rate:
            x = F.dropout(x,
                          p=float(self.dropout_rate),
                          training=self.training)
        return x if pre_logits else self.head(x)


class TimmVitFeaturesWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.head = None

    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, pre_logits=True)
        return x


class TimmVitPartialBlocks(torch.nn.Module):

    def __init__(self, model, block_idx):
        super().__init__()
        assert_block_idx(block_idx, model)
        self.blocks = model.blocks[:block_idx]
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed

        last_block = model.blocks[block_idx]
        self.attn = last_block.attn
        self.gamma_1 = last_block.gamma_1
        self.norm1 = last_block.norm1
        self.drop_path = last_block.drop_path


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x + self.pos_embed

        x = torch.cat((cls_tokens, x), dim=1)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        return x[:, 0]  # only cls token to save space by 197 times


class ResNetPartialBlocks(torch.nn.Module):

    def __init__(self, model, block_id, is_classifier=True):
        super().__init__()
        blocks = torch.nn.ModuleList()
        for i, layer in enumerate(model.children()):
            blocks.append(layer)

        self.is_classifier = is_classifier

        if self.is_classifier:
            self.blocks = blocks[block_id:]
        else:
            self.blocks = blocks[:block_id]

    def forward(self, x):

        for i, block in enumerate(self.blocks):

            if self.is_classifier and i + 1 == len(self.blocks):
                x = x.squeeze(-1).squeeze(-1)
            x = block(x)

        return x


def separate_classifier_and_featurizer(
    original_model,
    block_id=None,
    copy_model=True
):

    def model_modified_exception():
        raise Exception("Model modified by separate_classifier_and_featurizer")

    def is_timm_vit(model):
        return isinstance(
            model,
            (
                timm.models.vision_transformer.VisionTransformer,
                vit_models
            )
        )

    if copy_model:
        model = copy.deepcopy(original_model)
    else:
        model = original_model

    if block_id is not None:
        assert (
                is_timm_vit(model)
            or
                isinstance(model, torchvision.models.resnet.ResNet)
        )

    if is_mlp(model):
        featurizer = model
        classifier = make_dummy_object()
        classifier.in_features = featurizer.out_features
    elif is_timm_vit(model):
        if block_id is not None:
            classifier = TimmVitFinalLayers(model, block_id)
            featurizer = TimmVitPartialBlocks(model, block_id)
            # model.forward = lambda x: model_modified_exception()
        else:
            classifier = model.head
            featurizer = TimmVitFeaturesWrapper(model)
            model.forward = lambda x: model_modified_exception()
    elif isinstance(model, torchvision.models.resnet.ResNet):
        if block_id is not None:
            classifier = ResNetPartialBlocks(model, block_id, is_classifier=True)
            featurizer = ResNetPartialBlocks(model, block_id, is_classifier=False)
        else:
            assert copy_model
            classifier = model.fc
            featurizer = model
            featurizer.fc = torch.nn.Identity()
    else:
        raise_unknown(
            "model type",
            type(model),
            "separate_classifier_and_featurizer"
        )
    return featurizer, classifier


def wrap_model(model, wrapper_type):

    if wrapper_type is None:
        return model

    if is_ensemble(model):
        model = copy.deepcopy(model)
        for i in range(len(model.submodels)):
            model.submodels[i] = wrap_model(
                model.submodels[i],
                wrapper_type
            )
        if hasattr(model, "soup") and model.soup is not None:
            model.soup = wrap_model(model.soup, wrapper_type)
    else:
        if wrapper_type == "mvh":
            model = make_mvh_wrapper(model)
        elif wrapper_type == "IN9":
            model = make_in9_wrapper(model)
        elif wrapper_type == "ina":
            model = make_ina_wrapper(model)
        elif wrapper_type == "inr":
            model = make_inr_wrapper(model)
        else:
            raise_unknown("wrapper type", wrapper_type, "wrapper config")

    return model
