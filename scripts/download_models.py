import os
import sys
from stuned.utility.utils import (
    get_project_root_path,
)


# local imports
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(__file__))
)
from diverse_universe.local_datasets.utils import (
    SHARING_SUFFIX,
    download_file
)
sys.path.pop(0)


MODELS_FOLDER = os.path.join(get_project_root_path(), "models", "symlinked")
MODELS_DICT = {
    os.path.join(MODELS_FOLDER, "50_models.pkl"):
        "https://drive.google.com/file/d/1A9CgaeL78FLV2w_Vprzu0PYZ_EVLEJkC",
    os.path.join(MODELS_FOLDER, "ood_det_cov.pkl"):
        "https://drive.google.com/file/d/16ACrOLiano6Yw2N47trTgukOxn-0m6Z1",
    os.path.join(MODELS_FOLDER, "ood_det_sem.pkl"):
        "https://drive.google.com/file/d/18PcSJB8N31aMXzv34mmXO1ek2ixqGJGc",
    os.path.join(MODELS_FOLDER, "ood_gen.pkl"):
        "https://drive.google.com/file/d/1N_3feoS-0Fy_gPlu9KyiRAB7jbUFlDvL",
}


def main():
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    for model_path, model_url in MODELS_DICT.items():
        assert not os.path.exists(model_path), \
            f"Path {model_path} already exists!"
        download_file(model_path, model_url + SHARING_SUFFIX)
    print("All models downloaded!")


if __name__ == "__main__":
    main()
