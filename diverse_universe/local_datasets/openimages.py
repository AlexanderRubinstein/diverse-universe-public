import PIL
import sys
import os
import torch
# import shutil
from stuned.utility.utils import (
    get_project_root_path,
    get_with_assert
)
from stuned.local_datasets.transforms import (
    make_default_test_transforms_imagenet
)


# local modules
sys.path.insert(0, get_project_root_path())
from diverse_universe.local_datasets.utils import (
    ood_detection_only_warning,
    # download_tar_from_gdrive
    download_and_extract_tar
)
sys.path.pop(0)


OI_URL = "https://drive.google.com/uc?id=1NrqaSV9GFa33c0d-6N0qMxdy_qAqU6_7"


# TODO(Alex | 24.03.2024): drop requirement for train batch size and transform


# utility code below is taken from: https://github.com/haoqiwang/vim/blob/dabf9e5b242dbd31c15e09ff12af3d11f009f79c/list_dataset.py
def default_loader(path):
    return PIL.Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """flist format: impath label\nimpath label\n."""
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            data = line.strip().rsplit(maxsplit=1)
            if len(data) == 2:
                impath, imlabel = data
            else:
                impath, imlabel = data[0], 0
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(torch.utils.data.Dataset):

    def __init__(self,
                 root,
                 flist,
                 transform=None,
                 target_transform=None,
                 flist_reader=default_flist_reader,
                 loader=default_loader):
        self.root = root
        if flist is None:
            self.imlist = [(path, 0) for path in os.listdir(root)]
        else:
            self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def download_oi(data_dir):
    # download_tar_from_gdrive(
    download_and_extract_tar(
        data_dir,
        OI_URL,
        "openimages"
    ) # ??change url
    # shutil.move(
    #     os.path.join(os.path.dirname(data_dir), "test_filtered"),
    #     os.path.join(os.path.dirname(data_dir), "openimages")
    # )


def get_openimages_dataloader(
    eval_batch_size,
    # easy_robust_config,
    oi_config,
    num_workers,
    eval_transform,
    logger
):
    if eval_transform is None:
        eval_transform = make_default_test_transforms_imagenet()

    img_list = get_with_assert(oi_config, "img_list")
    # assert img_list is not None
    data_dir = get_with_assert(oi_config, "data_dir")
    dataset = ImageFilelist(data_dir, img_list, eval_transform)
    ood_detection_only_warning(logger)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dataloader
