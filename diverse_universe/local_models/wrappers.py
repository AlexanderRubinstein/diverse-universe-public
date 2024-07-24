import sys
# import os
# import torch
from stuned.utility.utils import (
    get_project_root_path,
    # extract_list_from_huge_string,
    # get_with_assert,
    add_custom_properties
)


# local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, get_project_root_path())
import external_libs.model_vs_human as mvh
# from utility.utils import (
#     add_custom_properties
# )
from diverse_universe.local_models.utils import (
    make_model_classes_wrapper,
    make_to_classes_mapping
)
sys.path.pop(0)


def make_mvh_model_wrapper(model):
    return make_model_classes_wrapper(model, make_mvh_mapper)


def make_mvh_mapper():
    human_categories = mvh.mapper.get_human_categories()

    mapper = make_to_classes_mapping(
        human_categories.get_imagenet_indices_for_category
    )

    mapper.categories = mvh.mapper.get_human_object_recognition_categories()

    mapper.category_to_index = {
        category: i for i, category in enumerate(mapper.categories)
    }

    return mapper
