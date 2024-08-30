# import h5py
import os
import sys
import torch
# import numpy as np
# import random
import torchvision
# from datasets import load_dataset
# import PIL
from stuned.utility.utils import (
    get_project_root_path,
    get_with_assert,
    # NAME_NUMBER_SEPARATOR,
    raise_unknown,
    # runcmd
    run_cmd_through_popen,
    remove_file_or_folder
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
from diverse_universe.local_datasets.utils import (
    ood_detection_only_warning
)
sys.path.pop(0)


def get_from_folder_dataloader(
    # train_batch_size,
    eval_batch_size,
    dataset_config,
    num_workers,
    eval_transform,
    logger
):

    data_dir = get_with_assert(dataset_config, "data_dir")
    dataset_name = get_with_assert(dataset_config, "dataset_name")
    assert dataset_name in data_dir, "Path does not contain dataset_name"
    if len(os.listdir(data_dir)) == 1:
        ood_detection_only_warning(logger)

    dataset = torchvision.datasets.ImageFolder(data_dir, eval_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # return {dataset_name: dataloader}
    return dataloader
