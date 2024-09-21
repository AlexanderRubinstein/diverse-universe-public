# import h5py
# import os
import sys
# import torch
# import numpy as np
# import random
# import torchvision
# from datasets import load_dataset
# import PIL


from stuned.utility.utils import (
    get_project_root_path,
    get_with_assert,
    # NAME_NUMBER_SEPARATOR,
    raise_unknown,
    # runcmd
    run_cmd_through_popen,
    remove_file_or_folder,
    log_or_print
    # get_hash,
    # parse_name_and_number
)
from stuned.local_datasets.transforms import (
    make_default_test_transforms_imagenet
)


# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, get_project_root_path())
# from local_datasets.utils import (
#     JSON_PATH,
#     get_generic_train_eval_dataloaders,
#     extract_subset_indices
# )
# from local_datasets.transforms import (
#     make_default_test_transforms_imagenet
# )
# from utility.utils import (
#     get_with_assert,
#     get_hash,
#     log_or_print,
#     error_or_print,
#     raise_unknown,
#     get_project_root_path,
#     read_json,
#     find_by_subkey
# )
# from local_models.utils import (
#     make_model_classes_wrapper
# )
from diverse_universe.local_datasets.from_folder import (
    get_from_folder_dataloader
)
from diverse_universe.local_datasets.utils import (
    # ood_detection_only_warning,
    # download_tar_from_gdrive
    download_and_extract_tar
)
sys.path.pop(0)


# taken from: https://github.com/deeplearning-wisc/large_scale_ood
INATURALIST_URL = "http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz"


# ??
def download_inat(data_dir, logger=None):
    log_or_print(
        "Downloading iNaturalist dataset (3.68GB) it can take up to 10 minutes",
        logger
    )
    # wget http://pages.cs.wisc.edu/~huangrui/imagenet_ood_dataset/iNaturalist.tar.gz
    # download_tar_from_gdrive("inaturalist", data_dir)
    download_and_extract_tar(data_dir, INATURALIST_URL, "iNaturalist")


def get_inat_dataloader(
    eval_batch_size,
    inat_config,
    num_workers,
    eval_transform,
    logger
):
    if eval_transform is None:
        eval_transform = make_default_test_transforms_imagenet()
    # ood_detection_only_warning(logger)
    return get_from_folder_dataloader(
        # train_batch_size,
        eval_batch_size=eval_batch_size,
        dataset_config=inat_config,
        num_workers=num_workers,
        eval_transform=eval_transform,
        logger=logger
    )
