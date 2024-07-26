import sys
import os
import torch
import copy
from stuned.utility.utils import (
    get_project_root_path,
    # extract_list_from_huge_string,
    # get_with_assert,
    # add_custom_properties
    raise_unknown,
    read_json
)
from stuned.utility.imports import lazy_import
# from modelvshuman.helper.human_categories import (
#     HumanCategories,
#     get_human_object_recognition_categories
# )


# lazy imports
mvh = lazy_import("modelvshuman")


# local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, get_project_root_path())
# from external_libs.model_vs_human.mapper as mvh
# from diverse_universe.external_libs.model_vs_human.mapper import (
#     get_human_categories,
#     get_human_object_recognition_categories
# )
# from diverse_universe.local_models.ensemble import (
#     # REDNECK_ENSEMBLE_KEY,
#     # SINGLE_MODEL_KEY,
#     # POE_KEY,
#     is_ensemble,
#     # make_redneck_ensemble,
#     # split_linear_layer
# )
# from utility.utils import (
#     add_custom_properties
# )
from diverse_universe.local_models.utils import (
    make_model_classes_wrapper,
    make_to_classes_mapping
)
from diverse_universe.local_datasets.utils import (
    JSON_PATH
)
sys.path.pop(0)


IN9_CATEGOREIS = tuple([
    "dog",
    "bird",
    "wheeled vehicle",
    "reptile",
    "carnivore",
    "insect",
    "musical instrument",
    "primate",
    "fish"
])
IN9_MAPPING_PATH = os.path.join(
    JSON_PATH,
    "in_to_in9.json"
)
IMAGENET_A_JSON = os.path.join(JSON_PATH, "imagenet_a_wnids.json")
IMAGENET_R_JSON = os.path.join(JSON_PATH, "imagenet_r_wnids.json")


def make_mvh_wrapper(model):
    return make_model_classes_wrapper(model, make_mvh_mapper)


def get_human_categories():
    return mvh.helper.human_categories.HumanCategories()


def make_mvh_mapper():
    human_categories = get_human_categories()

    mapper = make_to_classes_mapping(
        human_categories.get_imagenet_indices_for_category
    )

    mapper.categories = \
        mvh.helper.human_categories.get_human_object_recognition_categories()

    mapper.category_to_index = {
        category: i for i, category in enumerate(mapper.categories)
    }

    return mapper


def invert_dict_with_repetitions(d):
    res = {}
    for k, v in d.items():
        res.setdefault(v, [])
        res[v].append(k)
    return res


class IN9Categories:

    def __init__(self):
        map_to_in9 = read_json(IN9_MAPPING_PATH)
        self.in1000_to_in9 = invert_dict_with_repetitions(map_to_in9)
        self.in1000_to_in9 = {
            k: [int(vi) for vi in v] for k, v in self.in1000_to_in9.items()
        }

    def __call__(self, category_id):
        return self.in1000_to_in9[category_id]


def make_in9_mapper():
    mapper = make_to_classes_mapping(
        IN9Categories()
    )
    mapper.categories = list(range(len(IN9_CATEGOREIS)))
    return mapper


def make_in9_wrapper(model):
    return make_model_classes_wrapper(model, make_in9_mapper)


class ImagenetARMapper:

    def __init__(self, mapper_type, argmax_before=True):

        if mapper_type == "ina":
            json = read_json(IMAGENET_A_JSON)
            key = "a"
        else:
            assert mapper_type == "inr"
            key = "r"
            json = read_json(IMAGENET_R_JSON)

        self.imagenet_ar_mask = [
            wnid in json[f"imagenet_{key}_wnids"]
            for wnid
            in json["all_wnids"]
        ]
        self.argmax_before = argmax_before

    def __call__(self, probabilities):

        if self.argmax_before:
            argmax = probabilities.argmax(-1)
            probabilities = torch.zeros_like(probabilities)
            probabilities[torch.arange(probabilities.shape[0]), argmax] = 1

        return probabilities[:, self.imagenet_ar_mask]


def make_imagenet_ar_mapper(mapper_type):

    def make_mapper():
        return ImagenetARMapper(mapper_type)

    return make_mapper


def make_ina_wrapper(model):
    return make_model_classes_wrapper(model, make_imagenet_ar_mapper("ina"))


def make_inr_wrapper(model):
    return make_model_classes_wrapper(model, make_imagenet_ar_mapper("inr"))


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
