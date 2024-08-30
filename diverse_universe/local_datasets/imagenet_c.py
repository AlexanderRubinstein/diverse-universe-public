import torch
import sys
import torchvision
import os
from tqdm import tqdm
from stuned.utility.utils import (
    # get_project_root_path,
    find_by_subkey,
    get_with_assert,
    log_or_print
)
from stuned.local_datasets.transforms import (
    make_default_test_transforms_imagenet
)


# # local modules
# sys.path.insert(0, get_project_root_path())
# sys.path.pop(0)


IN_C_DATALOADERS_NAMES = {
    'Brightness': 'brightness',
    'Contrast': 'contrast',
    'Defocus Blur': 'defocus_blur',
    'Elastic Transform': 'elastic_transform',
    'Fog': 'fog',
    'Frost': 'frost',
    'Gaussian Noise': 'gaussian_noise',
    'Glass Blur': 'glass_blur',
    'Impulse Noise': 'impulse_noise',
    'JPEG Compression': 'jpeg_compression',
    'Motion Blur': 'motion_blur',
    'Pixelate': 'pixelate',
    'Shot Noise': 'shot_noise',
    'Snow': 'snow',
    'Zoom Blur': 'zoom_blur'
}
IN_C_MAX_SEVERITY = 5
IN_C_URL = "http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_datasets/imagenet-c.tar.gz"


def extract_in_c_paths(base_path, strengths, filename_substring):
    all_corruptions = list(IN_C_DATALOADERS_NAMES.keys())
    res = {}
    for corruption in all_corruptions:
        for strength in strengths:
            path = os.path.join(base_path, f"{corruption}_{strength}")
            filenames = os.listdir(path)

            filename = find_by_subkey(
                filenames,
                filename_substring,
                assert_found=True,
                only_first_occurence=False
            )
            assert len(filename) == 1
            filename = filename[0]

            res[f"{IN_C_DATALOADERS_NAMES[corruption]}_{strength}"] \
                = os.path.join(path, filename)
    return res


# inspired by: https://github.com/alibaba/easyrobust/blob/71262215c368fd21cfd1476c8fa3ec4ece53459a/easyrobust/benchmarks/ood/imagenet_c.py
def get_imagenet_c_dataloaders(
    eval_batch_size,
    in_c_config,
    num_workers,
    eval_transform,
    logger,
    severity_levels=list(range(1, IN_C_MAX_SEVERITY + 1))
):

    inc_dataloaders = {}
    data_dir = get_with_assert(in_c_config, "data_dir")
    inc_types = in_c_config.get("inc_types")
    if inc_types is not None:
        inc_types = set(inc_types)
    if eval_transform is None:
        eval_transform = make_default_test_transforms_imagenet()
    log_or_print("Creating ImageNet-C dataloaders", logger)
    for name, subdir in tqdm(IN_C_DATALOADERS_NAMES.items()):
        if inc_types is not None and name not in inc_types:
            continue
        for severity in severity_levels:
            severity = str(severity)
            inc_dataset = torchvision.datasets.ImageFolder(
                os.path.join(
                    data_dir,
                    subdir,
                    severity
                ),
                transform=eval_transform
            )
            inc_dataloaders[name + '_' + severity] = torch.utils.data.DataLoader(
                inc_dataset,
                sampler=None,
                batch_size=eval_batch_size,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False
            )
    return inc_dataloaders
