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
    # remove_file_or_folder
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
    download_and_extract_tar
)
sys.path.pop(0)


INA_FINGERPRINT = 76
INR_FINGERPRINT = 103
BASE_EASY_ROBUST_URL = "http://alisec-competition.oss-cn-shanghai.aliyuncs.com/xiaofeng/easy_robust/benchmark_datasets/"
IN_A_URL = BASE_EASY_ROBUST_URL + "imagenet-a.tar.gz"
IN_R_URL = BASE_EASY_ROBUST_URL + "imagenet-r.tar.gz"


# IN_C_DATALOADERS_NAMES = {
#     'Brightness': 'brightness',
#     'Contrast': 'contrast',
#     'Defocus Blur': 'defocus_blur',
#     'Elastic Transform': 'elastic_transform',
#     'Fog': 'fog',
#     'Frost': 'frost',
#     'Gaussian Noise': 'gaussian_noise',
#     'Glass Blur': 'glass_blur',
#     'Impulse Noise': 'impulse_noise',
#     'JPEG Compression': 'jpeg_compression',
#     'Motion Blur': 'motion_blur',
#     'Pixelate': 'pixelate',
#     'Shot Noise': 'shot_noise',
#     'Snow': 'snow',
#     'Zoom Blur': 'zoom_blur'
# }
# IN_C_MAX_SEVERITY = 5


def get_easy_robust_dataloaders(
    train_batch_size,
    eval_batch_size,
    easy_robust_config,
    num_workers,
    eval_transform,
    logger
):
    dataset_types = get_with_assert(easy_robust_config, "dataset_types")

    val_dataloaders = {}

    if eval_transform is None:
        eval_transform = make_default_test_transforms_imagenet()

    for dataset_type in dataset_types:
        assert dataset_type not in val_dataloaders, "Duplicate dataset type"
        if dataset_type in ["imagenet_a", "imagenet_r", "imagenet_v2"]:
            val_dataloaders[dataset_type] = get_imagenet_arv2_dataloader(
                # train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                easyrobust_config=easy_robust_config,
                num_workers=num_workers,
                eval_transform=eval_transform,
                logger=logger,
                dataset_type=dataset_type
            )
        # elif dataset_type == "imagenet_hard":
        #     val_dataloaders[dataset_type] = get_imagenet_hard_dataloader(
        #         train_batch_size=train_batch_size,
        #         eval_batch_size=eval_batch_size,
        #         easyrobust_config=easy_robust_config,
        #         num_workers=num_workers,
        #         eval_transform=eval_transform,
        #         logger=logger
        #     )
        # elif dataset_type == "imagenet_c":
        #     val_dataloaders |= get_imagenet_c_dataloader(
        #         train_batch_size,
        #         eval_batch_size,
        #         easy_robust_config,
        #         num_workers,
        #         eval_transform,
        #         logger
        #     )
        # elif dataset_type == "openimages":
        #     val_dataloaders[dataset_type] = get_openimages_dataloader(
        #         train_batch_size,
        #         eval_batch_size,
        #         easy_robust_config,
        #         num_workers,
        #         eval_transform,
        #         logger
        #     )
        # elif dataset_type == "from_folder":
        #     val_dataloaders |= get_from_folder_dataloader(
        #         train_batch_size,
        #         eval_batch_size,
        #         easy_robust_config,
        #         num_workers,
        #         eval_transform,
        #         logger
        #     )
        # elif dataset_type == "imagenet_d":
        #     val_dataloaders |= get_imagenet_d_dataloaders(
        #         train_batch_size,
        #         eval_batch_size,
        #         easy_robust_config,
        #         num_workers,
        #         eval_transform,
        #         logger
        #     )
        else:
            raise_unknown(
                "dataset type",
                dataset_type,
                "easy_robust_config"
            )
    return None, val_dataloaders


# based on: https://github.com/alibaba/easyrobust/blob/main/easyrobust/benchmarks/ood/imagenet_a.py#L12
def get_imagenet_arv2_dataloader(
    eval_batch_size,
    easyrobust_config,
    num_workers,
    eval_transform,
    logger,
    dataset_type
):

    data_dir = get_with_assert(easyrobust_config, "data_dir")

    if dataset_type == "imagenet_v2":
        assert os.path.exists(os.path.join(data_dir, "0000"))
    else:
        fingerprint_dir = os.path.join(
            data_dir,
            "n01498041"
        )
        fingerprint = len(os.listdir(fingerprint_dir))
        if dataset_type == "imagenet_a":
            assert fingerprint == INA_FINGERPRINT, "Unexpected fingerprint"
        else:
            assert dataset_type == "imagenet_r"
            assert fingerprint == INR_FINGERPRINT, "Unexpected fingerprint"

    dataset_ina = torchvision.datasets.ImageFolder(
        data_dir,
        transform=eval_transform
    )
    sampler = None
    return torch.utils.data.DataLoader(
        dataset_ina,
        sampler=sampler,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )


# def extract_tar(tar_path):
#     run_cmd_through_popen(
#         f"tar -zxf {tar_path} ",
#         # verbose=True,
#         logger=None
#     )


# def download_oi(path):
#     def make_extract_cmd(file):
#         return (
#             f"tar -zxf {file} " # -C $FOLDER
#             f"&& rm {file}"
#         )

#     downloaded_tar = None # ??
#     # gdown(test)
#     # gdown list
#     extract_tar(downloaded_tar)
#     remove_file_or_folder(downloaded_tar)


# def make_download_cmd(name, path, url):
#     return (
#         f"export FOLDER={path} "
#         f"&& mkdir -p $FOLDER "
#         f"&& export FILE=$FOLDER/{name}.tar.gz "
#         f"&& wget {url} -O $FILE && tar -zxf $FILE -C $FOLDER "
#         f"&& rm $FOLDER/{name}.tar.gz"
#     )


def download_in_a(path):
    #         runcmd(
#             f"cd {data_path} && rm {archive_path}",
#             verbose=True,
#             logger=logger
#         )
# run_cmd_through_popen(cmd_to_run, logger)
    # run_cmd_through_popen(
    #     make_download_cmd("imagenet-a", path, IN_A_URL),
    #     # verbose=True,
    #     logger=None
    # )
    # download_and_extract_tar("imagenet-r", data_dir, download_url)
    # os.sytem(
    #     make_download_cmd("imagenet-a", path, IN_A_URL)
    # )
    download_and_extract_tar(path, IN_A_URL, "imagenet-a")


def download_in_r(path):
    # run_cmd_through_popen(
    #     make_download_cmd("imagenet-r", path, IN_R_URL),
    #     # verbose=True,
    #     logger=None
    # )
    # os.sytem(
    #     make_download_cmd("imagenet-r", path, IN_R_URL)
    # )
    download_and_extract_tar(path, IN_R_URL, "imagenet-r")


# def collate_fn_hard(batch):
#     labels = [item["label"] for item in batch]
#     labels = [label + [-1] * (10 - len(label)) for label in labels]

#     return torch.stack([item["image"]
#                         for item in batch]), torch.tensor(labels)

# # based on: https://github.com/kirill-vish/Beyond-INet/blob/fd9b1b6c36ecf702fbcc355e037d8e9d307b0137/inference/robustness.py#L72
# def get_imagenet_hard_dataloader(
#     train_batch_size,
#     eval_batch_size,
#     easyrobust_config,
#     num_workers,
#     eval_transform,
#     logger
# ):

#     data_dir = get_with_assert(easyrobust_config, "data_dir")

#     # https://huggingface.co/datasets/taesiri/imagenet-hard
#     dataset_val = load_dataset("taesiri/imagenet-hard",
#                                split="validation",
#                                cache_dir=data_dir)

#     transform = eval_transform
#     if transform is None:
#         transform = make_default_test_transforms_imagenet()

#     def apply_transforms(examples):
#         examples["pixel_values"] = examples["image"]
#         examples["image"] = [transform(image) for image in examples["image"]]
#         return examples

#     dataset_val.set_transform(apply_transforms)

#     data_loader = torch.utils.data.DataLoader(dataset_val,
#                                               batch_size=eval_batch_size,
#                                               num_workers=num_workers,
#                                               pin_memory=True,
#                                               shuffle=False,
#                                               collate_fn=collate_fn_hard)

#     return data_loader


# # inspired by: https://github.com/alibaba/easyrobust/blob/71262215c368fd21cfd1476c8fa3ec4ece53459a/easyrobust/benchmarks/ood/imagenet_c.py
# def get_imagenet_c_dataloader(
#     train_batch_size,
#     eval_batch_size,
#     easy_robust_config,
#     num_workers,
#     eval_transform,
#     logger
# ):

#     inc_dataloaders = {}
#     data_dir = get_with_assert(easy_robust_config, "data_dir")
#     inc_types = easy_robust_config.get("inc_types")
#     if inc_types is not None:
#         inc_types = set(inc_types)
#     for name, subdir in IN_C_DATALOADERS_NAMES.items():
#         if inc_types is not None and name not in inc_types:
#             continue
#         for severity in range(1, IN_C_MAX_SEVERITY + 1):
#             severity = str(severity)
#             inc_dataset = torchvision.datasets.ImageFolder(
#                 os.path.join(
#                     data_dir,
#                     subdir,
#                     severity
#                 ),
#                 transform=eval_transform
#             )
#             inc_dataloaders[name + '_' + severity] = torch.utils.data.DataLoader(
#                 inc_dataset,
#                 sampler=None,
#                 batch_size=eval_batch_size,
#                 num_workers=num_workers,
#                 pin_memory=True,
#                 drop_last=False
#             )
#     return inc_dataloaders


# # TODO(Alex | 24.03.2024): OpenImages is not from easy robust, move it to a separate file
# # TODO(Alex | 24.03.2024): drop requirement for train batch size and transform


# # utility code below is taken from: https://github.com/haoqiwang/vim/blob/dabf9e5b242dbd31c15e09ff12af3d11f009f79c/list_dataset.py
# def default_loader(path):
#     return PIL.Image.open(path).convert('RGB')


# def default_flist_reader(flist):
#     """flist format: impath label\nimpath label\n."""
#     imlist = []
#     with open(flist, 'r') as rf:
#         for line in rf.readlines():
#             data = line.strip().rsplit(maxsplit=1)
#             if len(data) == 2:
#                 impath, imlabel = data
#             else:
#                 impath, imlabel = data[0], 0
#             imlist.append((impath, int(imlabel)))

#     return imlist


# class ImageFilelist(torch.utils.data.Dataset):

#     def __init__(self,
#                  root,
#                  flist,
#                  transform=None,
#                  target_transform=None,
#                  flist_reader=default_flist_reader,
#                  loader=default_loader):
#         self.root = root
#         self.imlist = flist_reader(flist)
#         self.transform = transform
#         self.target_transform = target_transform
#         self.loader = loader

#     def __getitem__(self, index):
#         impath, target = self.imlist[index]
#         img = self.loader(os.path.join(self.root, impath))
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target

#     def __len__(self):
#         return len(self.imlist)


# def get_openimages_dataloader(
#     train_batch_size,
#     eval_batch_size,
#     easy_robust_config,
#     num_workers,
#     eval_transform,
#     logger
# ):
#     img_list = get_with_assert(easy_robust_config, "img_list")
#     assert img_list is not None
#     data_dir = get_with_assert(easy_robust_config, "data_dir")
#     dataset = ImageFilelist(data_dir, img_list, eval_transform)
#     ood_detection_only_warning(logger)

#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=eval_batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False
#     )

#     return dataloader


# def ood_detection_only_warning(logger):
#     logger.error(
#         "All images have label 0 here. "
#         "This dataloader is purely for OOD detection."
#     )


# def get_from_folder_dataloader(
#     train_batch_size,
#     eval_batch_size,
#     easy_robust_config,
#     num_workers,
#     eval_transform,
#     logger
# ):

#     data_dir = get_with_assert(easy_robust_config, "data_dir")
#     dataset_name = get_with_assert(easy_robust_config, "dataset_name")
#     assert dataset_name in data_dir, "Path does not contain dataset_name"
#     if len(os.listdir(data_dir)) == 1:
#         ood_detection_only_warning(logger)

#     dataset = torchvision.datasets.ImageFolder(data_dir, eval_transform)

#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=eval_batch_size,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=True,
#         drop_last=False
#     )

#     return {dataset_name: dataloader}


# def extract_in_c_paths(base_path, strengths, filename_substring):
#     all_corruptions = list(IN_C_DATALOADERS_NAMES.keys())
#     res = {}
#     for corruption in all_corruptions:
#         for strength in strengths:
#             path = os.path.join(base_path, f"{corruption}_{strength}")
#             filenames = os.listdir(path)

#             filename = find_by_subkey(
#                 filenames,
#                 filename_substring,
#                 assert_found=True,
#                 only_first_occurence=False
#             )
#             assert len(filename) == 1
#             filename = filename[0]

#             res[f"{IN_C_DATALOADERS_NAMES[corruption]}_{strength}"] \
#                 = os.path.join(path, filename)
#     return res


# # TODO(Alex | 24.04.2024): ImageNet-D is not from easy robust, move it to a separate file


# import json


# IMAGENET_D_SUBSETS = ['background', 'texture', 'material']
# IMAGENET_D_ID_MAP_JSON = os.path.join(JSON_PATH, "imgnet_d2imgnet_id.json")


# # taken from: https://github.com/chenshuang-zhang/imagenet_d/blob/main/utils/data_loaders_imgnet_id.py#L6
# class ImageNetDLoader(torch.utils.data.Dataset):

#     def __init__ (self,
#                   test_base_dir, few_test=None, transform=None, center_crop=False
#                   ):
#         super().__init__()

#         self.test_path = test_base_dir
#         self.categories_list = os.listdir(self.test_path)
#         self.categories_list.sort()

#         self.file_lists = []
#         self.label_lists = []
#         self.few_test = few_test

#         self.transforms=transform

#         with open(IMAGENET_D_ID_MAP_JSON) as f:
#             self.dict_imgnet_d2imagenet_id = json.load(f)

#         for each in self.categories_list:
#             folder_path = os.path.join(self.test_path, each)

#             files_names = os.listdir(folder_path)

#             for eachfile in files_names:
#                 image_path = os.path.join(folder_path, eachfile)
#                 self.file_lists.append(image_path)
#                 self.label_lists.append(self.dict_imgnet_d2imagenet_id[each]+[-1]*(10-len(self.dict_imgnet_d2imagenet_id[each])))

#     def __len__(self):
#         if self.few_test is not None:
#             return self.few_test
#         else:
#             return len(self.label_lists)

#     def _transform(self, sample):
#         return self.transforms(sample)

#     def __getitem__(self, item):
#         path_list=self.file_lists[item]
#         img = PIL.Image.open(path_list).convert("RGB")

#         img_tensor = self._transform(img)
#         img.close()
#         labels = self.label_lists[item]

#         return {"images": img_tensor, "labels": labels, "path": path_list}


# def get_collate_fn_in_d(drop_paths):
#     def collate_fn_in_d(examples):
#         images = []
#         labels = []
#         paths = []
#         for example in examples:
#             images.append(example["images"])
#             # we take only first label as we are not interested in top-5 accuracy
#             labels.append(torch.tensor(example["labels"][0], dtype=torch.long))
#             paths.append(example["path"])
#         if drop_paths:
#             return torch.stack(images), torch.stack(labels)
#         else:
#             return torch.stack(images), torch.stack(labels), paths

#     return collate_fn_in_d


# def get_imagenet_d_dataloaders(
#     train_batch_size,
#     eval_batch_size,
#     easy_robust_config,
#     num_workers,
#     eval_transform,
#     logger
# ):

#     inc_dataloaders = {}
#     data_dir = get_with_assert(easy_robust_config, "data_dir")
#     ind_types = easy_robust_config.get("ind_types")
#     if ind_types is not None:
#         ind_types = set(ind_types)
#     ind_dataloaders = {}
#     for dataloader_name in IMAGENET_D_SUBSETS:
#         if ind_types is not None and dataloader_name not in ind_types:
#             continue
#         ind_dataloaders[dataloader_name] = torch.utils.data.DataLoader(
#             ImageNetDLoader(
#                 os.path.join(data_dir, dataloader_name),
#                 transform=eval_transform
#             ),
#             batch_size=eval_batch_size,
#             shuffle=True,
#             num_workers=num_workers,
#             pin_memory=True,
#             collate_fn=get_collate_fn_in_d(drop_paths=True)
#         )
#     return ind_dataloaders
