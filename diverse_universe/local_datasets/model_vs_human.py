# import sys
# import os
import torch
# import torchvision.datasets as datasets
import os
import torchvision
import sys
# import torch
from stuned.utility.utils import (
    get_project_root_path,
    read_json,
    # NAME_NUMBER_SEPARATOR,
    # get_with_assert,
    # raise_unknown,
    # get_hash,
    # parse_name_and_number
)
from stuned.utility.imports import lazy_import
from stuned.local_datasets.utils import (
    wrap_dataloader
)
from stuned.local_datasets.transforms import (
    make_normalization_transforms
)
# from modelvshuman.utils import (
#     load_dataset
# )


# lazy imports
mvh = lazy_import("modelvshuman")


# local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(
    0,
    get_project_root_path()
)
# from src.utility.utils import (
#     get_project_root_path,
#     read_json
# )
# from src.local_datasets.utils import (
#     wrap_dataloader
# )
# from src.local_datasets.transforms import (
#     DEFAULT_MEAN_IN,
#     DEFAULT_STD_IN,
#     make_normalization_transforms
# )
# from src.local_models.model_vs_human import (
#     make_mvh_mapper
# )
from diverse_universe.local_datasets.utils import (
    JSON_PATH
)
from diverse_universe.local_models.wrappers import (
    make_mvh_mapper
)
# import external_libs.model_vs_human as mvh
sys.path.pop(0)


# DEFAULT_MEAN = DEFAULT_MEAN_IN
# DEFAULT_STD = DEFAULT_STD_IN
TRUE_LABELS_JSON_PATH = os.path.join(
    JSON_PATH,
    "cue_conflict_labels.json"
)


def get_modelvshuman_dataset(dataset_type, batch_size, num_workers=1):
    """

    Available dataset types are:

    [
        'imagenet_validation', 'sketch', 'stylized', 'original', 'greyscale',
        'texture', 'edge', 'silhouette', 'cue-conflict', 'colour', 'contrast',
        'high-pass', 'low-pass', 'phase-scrambling', 'power-equalisation',
        'false-colour', 'rotation', 'eidolonI', 'eidolonII', 'eidolonIII',
        'uniform-noise'
    ]

    """

    # mvh_utils = mvh.utils.import_from_model_vs_human("utils")

    dataset = mvh.utils.load_dataset(
        dataset_type,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return dataset


def make_modelvshuman_dataloader(
    dataset_type,
    batch_size,
    num_workers,
    to_wrap=True,
    to_prune=False
):

    dataset = get_modelvshuman_dataset(
        dataset_type,
        batch_size,
        num_workers
    )

    if to_prune:
        dataset = subset_cue_conflict(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    else:
        dataloader = dataset.loader

    if to_wrap:
        return wrap_mvh_dataloader(dataloader)

    return dataloader


def get_modelvshuman_dataloaders(
    mvh_config,
    train_batch_size,
    test_batch_size
):

    train_dataloader = None
    test_dataloaders = None

    num_workers = mvh_config.get("num_workers", 1)
    to_prune = mvh_config.get("to_prune", False)

    if train_batch_size > 0:
        train_type = mvh_config["train_type"]
        train_dataloader = make_modelvshuman_dataloader(
            train_type,
            train_batch_size,
            num_workers,
            to_wrap=True,
            to_prune=to_prune
        )

    if test_batch_size > 0:
        test_dataloaders = {}
        test_types = mvh_config["test_types"]
        assert isinstance(test_types, list)
        for test_type in test_types:
            test_dataloaders[test_type] = make_modelvshuman_dataloader(
                test_type,
                test_batch_size,
                num_workers,
                to_wrap=True,
                to_prune=to_prune
            )

    return train_dataloader, test_dataloaders


class ModelVsHumanIteratorWrapper:

    def __init__(self, iterator):
        self.iterator = iterator
        self.mapper = make_mvh_mapper()

    def __iter__(self):
        return self

    def __next__(self):
        inputs, labels, paths = next(self.iterator)
        if isinstance(labels, (list, tuple)):
            labels = torch.tensor(
                [self.mapper.category_to_index[category] for category in labels],
                dtype=torch.int64
            )
        return inputs, labels


def wrap_mvh_dataloader(dataloader):
    return wrap_dataloader(dataloader, wrapper=ModelVsHumanIteratorWrapper)


# def get_image_unnormalizer(mean=DEFAULT_MEAN, std=DEFAULT_STD):

#     simclr_utils = mvh.utils.import_from_model_vs_human("models.pytorch.simclr.utils")

#     return simclr_utils.modules.Unnormalize(
#         mean=mean,
#         std=std
#     )


def get_shape_texture(i, shape_texture_labels):
    gt_info = shape_texture_labels[str(i)]
    shape_gt = gt_info[2]
    texture_gt = gt_info[3]
    return shape_gt, texture_gt


def subset_cue_conflict(
    dataset,
    shape_texture_labels=None,
    shape_predictor_res=None,
    texture_predictor_res=None
):
    if shape_texture_labels is None:
        shape_texture_labels = read_json(TRUE_LABELS_JSON_PATH)

    return SubsetCC(
        dataset,
        shape_texture_labels,
        shape_predictor_res,
        texture_predictor_res
    )


class SubsetCC(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset,
        shape_texture_labels,
        shape_predictor_res,
        texture_predictor_res
    ):

        self.dataset = torchvision.datasets.ImageFolder(
            dataset.path,
            make_normalization_transforms()
        )
        self.active_indices = []
        self.shape_texture_labels = shape_texture_labels
        self.shape_predictor_res = shape_predictor_res
        self.texture_predictor_res = texture_predictor_res

        for i in range(len(self.shape_texture_labels)):

            shape_gt, texture_gt = get_shape_texture(
                i,
                self.shape_texture_labels
            )

            shape_pred = get_prediction(
                self.shape_predictor_res,
                i,
                shape_gt
            )

            texture_pred = get_prediction(
                self.texture_predictor_res,
                i,
                texture_gt
            )

            if (
                    shape_gt != texture_gt
                and
                    shape_pred == shape_gt
                and
                    texture_pred == texture_gt
            ):
                self.active_indices.append(i)

    def __getitem__(self, idx):

        i = self.active_indices[idx]
        shape_gt, texture_gt = get_shape_texture(i, self.shape_texture_labels)

        dataset_item = self.dataset[i]

        if self.shape_predictor_res is None:
            assert self.texture_predictor_res is None
            return (
                *dataset_item,
                shape_gt
            )
        else:
            shape_pred = self.shape_predictor_res[i]
            texture_pred = self.texture_predictor_res[i]

            return (
                *dataset_item,
                shape_gt,
                texture_gt,
                shape_pred,
                texture_pred
            )

    def __len__(self):
        return len(self.active_indices)


# def get_prediction(predictor_res, i, default):
#     if predictor_res is None:
#         return default
#     else:
#         return predictor_res[i]


# def max_by_key(cue_accs, key):
#     model_names = list(cue_accs.keys())
#     return max(model_names, key=lambda x: cue_accs[x][key])
