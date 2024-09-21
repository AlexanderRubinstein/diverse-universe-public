# import gdown
import os
import sys
# import shutil
# from tqdm import tqdm

from stuned.utility.utils import (
    get_project_root_path,
    # log_or_print,
    # read_yaml
)


# local imports
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(__file__))
)
# from diverse_universe.local_datasets.imagenet_c import (
#     IN_C_DATALOADERS_NAMES,
# )
# from diverse_universe.cross_eval.eval import (
#     extract_from_pkl
# )
from diverse_universe.local_datasets.utils import download_and_extract_tar
sys.path.pop(0)


# MODELS_FOLDER = os.path.join(get_project_root_path(), "models", "symlinked")
# MODELS_DICT = {
#     os.path.join(MODELS_FOLDER, "50_models.pkl"): " url ??",
#     os.path.join(MODELS_FOLDER, "ood_det_cov.pkl"): " url ??",
#     os.path.join(MODELS_FOLDER, "ood_det_sem.pkl"): " url ??",
#     os.path.join(MODELS_FOLDER, "ood_gen.pkl"): " url ??",
# }


CACHED_DATA_URL = "https://drive.google.com/file/d/1CL6U8oa7vCCrYdVCcTrRZSUFVKXEe3q0/view?usp=sharing"
DATASETS_FOLDER = os.path.join(
    get_project_root_path(),
    "data",
    "datasets",
    "cached"
)


def main():
    parent_folder = os.path.dirname(DATASETS_FOLDER)
    os.makedirs(parent_folder, exist_ok=True)
    # for model_path, model_url in MODELS_DICT.items():
    #     assert not os.path.exists(model_path), \
    #         f"Path {model_path} already exists!"
    #     download_file(model_path, model_url)
    download_and_extract_tar(DATASETS_FOLDER, CACHED_DATA_URL)
    # shutil.move(
    #     os.path.join(parent_folder, "cache"),
    #     os.path.join(DATASETS_FOLDER)
    # )


if __name__ == "__main__":
    main()
