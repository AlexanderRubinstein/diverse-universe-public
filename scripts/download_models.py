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
from diverse_universe.local_datasets.utils import download_file
sys.path.pop(0)


MODELS_FOLDER = os.path.join(get_project_root_path(), "models", "symlinked")
MODELS_DICT = {
    os.path.join(MODELS_FOLDER, "50_models.pkl"): "https://drive.google.com/uc?id=1qFmQ2t1gIegAUrXDcQKNa8wrTym0t6AK",
    os.path.join(MODELS_FOLDER, "ood_det_cov.pkl"): "https://drive.google.com/file/d/1FikmdPrxNfsTWg9s2uWGpilhZi7ApKzO/view?usp=drive_link",
    os.path.join(MODELS_FOLDER, "ood_det_sem.pkl"): "https://drive.google.com/file/d/1HBFgXMx3JU7XmVJzLhdL3hf0e9wbLNhS/view?usp=drive_link",
    os.path.join(MODELS_FOLDER, "ood_gen.pkl"): "https://drive.google.com/file/d/1Z1ILwX0yfvrdz-N0sTiqHsAcEDHrbe2K/view?usp=drive_link",
}


def main():
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    for model_path, model_url in MODELS_DICT.items():
        assert not os.path.exists(model_path), \
            f"Path {model_path} already exists!"
        download_file(model_path, model_url)


if __name__ == "__main__":
    main()