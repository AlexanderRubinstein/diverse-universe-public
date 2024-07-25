# import sys
# import os
# from stuned.utility.utils import (
#     get_project_root_path,
#     # NAME_NUMBER_SEPARATOR,
#     # get_with_assert,
#     # raise_unknown,
#     # get_hash,
#     # parse_name_and_number
# )


# # local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# from src.utility.utils import (
#     get_project_root_path
# )
# from src.utility.imports import import_from_string
# sys.path.pop(0)


# MVH_ENV_NAME = "MODELVSHUMANDIR"
# MODEL_VS_HUMAN_DIR = os.path.join(
#     get_project_root_path(),
#     "submodules",
#     "model-vs-human"
# )
# PYTORCH_MODELS_DIR = os.path.join(
#     MODEL_VS_HUMAN_DIR,
#     "modelvshuman",
#     "models",
#     "pytorch"
# )
# MODEL_VS_HUMAN_PKG = "submodules.model-vs-human.modelvshuman"


# def import_from_model_vs_human(module_as_str):
#     sys.path.insert(0, get_project_root_path())
#     # add models dir to path to see "clip"
#     sys.path.insert(
#         0,
#         PYTORCH_MODELS_DIR
#     )
#     os.environ[MVH_ENV_NAME] = MODEL_VS_HUMAN_DIR
#     module = import_from_string(
#         f"{MODEL_VS_HUMAN_PKG}.{module_as_str}"
#     )
#     sys.path.pop(0)
#     sys.path.pop(0)
#     os.environ.pop(MVH_ENV_NAME)
#     return module
