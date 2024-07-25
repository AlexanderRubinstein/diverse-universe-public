# from wilds import get_dataset
# from wilds.common.data_loaders import (
#     get_train_loader,
#     get_eval_loader
# )
# import sys
# import os
# import torchvision.transforms as transforms
# from wilds.common.grouper import CombinatorialGrouper
import torch
from stuned.local_datasets.utils import (
    # make_default_cache_path,
    # chain_dataloaders,
    # randomly_subsampled_dataloader,
    # # make_or_load_from_cache
    make_default_data_path
)


# # local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# from local_datasets.utils import (
#     DropLastIteratorWrapper,
#     make_default_data_path,
#     wrap_dataloader
# )
# from utility.helpers_for_tests import (
#     make_dummy_object
# )
# from utility.utils import (
#     get_project_root_path,
#     runcmd,
#     log_or_print
# )
# # TODO(Alex | 30.01.2024): Uncomment when dependency fixed
# # from external_libs.div_dis import (
# #     dataset_defaults
# # )
# sys.path.pop(0)


# WATERBIRDS_RES = (224, 224)
# SPLITS = ("train", "val", "test")
# WATERBIRDS_DIVDIS_LINK = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"


# # taken from wilds/configs/datasets.py in DivDis repo
# DATASET_DEFAULTS = {
#     "amazon": {
#         "split_scheme": "official",
#         "model": "distilbert-base-uncased",
#         "transform": "bert",
#         "max_token_length": 512,
#         "loss_function": "cross_entropy",
#         "algo_log_metric": "accuracy",
#         "batch_size": 8,
#         "unlabeled_batch_size": 8,
#         "lr": 1e-5,
#         "weight_decay": 0.01,
#         "n_epochs": 3,
#         "n_groups_per_batch": 2,
#         "unlabeled_n_groups_per_batch": 2,
#         "irm_lambda": 1.0,
#         "coral_penalty_weight": 1.0,
#         "dann_penalty_weight": 1.0,
#         "dann_featurizer_lr": 1e-6,
#         "dann_classifier_lr": 1e-5,
#         "dann_discriminator_lr": 1e-5,
#         "loader_kwargs": {
#             "num_workers": 1,
#             "pin_memory": True,
#         },
#         "unlabeled_loader_kwargs": {
#             "num_workers": 1,
#             "pin_memory": True,
#         },
#         "process_outputs_function": "multiclass_logits_to_pred",
#         "process_pseudolabels_function": "pseudolabel_multiclass_logits",
#     },
#     "bdd100k": {
#         "split_scheme": "official",
#         "model": "resnet50",
#         "model_kwargs": {"pretrained": True},
#         "loss_function": "multitask_bce",
#         "val_metric": "acc_all",
#         "val_metric_decreasing": False,
#         "optimizer": "SGD",
#         "optimizer_kwargs": {"momentum": 0.9},
#         "batch_size": 32,
#         "lr": 0.001,
#         "weight_decay": 0.0001,
#         "n_epochs": 10,
#         "algo_log_metric": "multitask_binary_accuracy",
#         "transform": "image_base",
#         "process_outputs_function": "binary_logits_to_pred",
#     },
#     "camelyon17": {
#         "split_scheme": "official",
#         "model": "densenet121",
#         "model_kwargs": {"pretrained": False},
#         "transform": "image_base",
#         "target_resolution": (96, 96),
#         "loss_function": "cross_entropy",
#         "groupby_fields": ["hospital"],
#         "val_metric": "acc_avg",
#         "val_metric_decreasing": False,
#         "optimizer": "SGD",
#         "optimizer_kwargs": {"momentum": 0.9},
#         "scheduler": None,
#         "batch_size": 32,
#         "unlabeled_batch_size": 32,
#         "lr": 0.001,
#         "weight_decay": 0.01,
#         "n_epochs": 10,
#         "n_groups_per_batch": 2,
#         "unlabeled_n_groups_per_batch": 2,
#         "irm_lambda": 1.0,
#         "coral_penalty_weight": 0.1,
#         "dann_penalty_weight": 0.1,
#         "dann_featurizer_lr": 0.0001,
#         "dann_classifier_lr": 0.001,
#         "dann_discriminator_lr": 0.001,
#         "algo_log_metric": "accuracy",
#         "process_outputs_function": "multiclass_logits_to_pred",
#         "process_pseudolabels_function": "pseudolabel_multiclass_logits",
#     },
#     "celebA": {
#         "split_scheme": "official",
#         "model": "resnet50",
#         "model_kwargs": {"pretrained": True},
#         "transform": "image_base",
#         "loss_function": "cross_entropy",
#         "groupby_fields": ["male", "y"],
#         "val_metric": "acc_wg",
#         "val_metric_decreasing": False,
#         "optimizer": "SGD",
#         "optimizer_kwargs": {"momentum": 0.9},
#         "scheduler": None,
#         "batch_size": 64,
#         "lr": 0.001,
#         "weight_decay": 0.0,
#         "n_epochs": 200,
#         "algo_log_metric": "accuracy",
#         "process_outputs_function": "multiclass_logits_to_pred",
#     },
#     "civilcomments": {
#         "split_scheme": "official",
#         "model": "distilbert-base-uncased",
#         "transform": "bert",
#         "loss_function": "cross_entropy",
#         "groupby_fields": ["black", "y"],
#         "val_metric": "acc_wg",
#         "val_metric_decreasing": False,
#         "batch_size": 16,
#         "unlabeled_batch_size": 16,
#         "lr": 1e-5,
#         "weight_decay": 0.01,
#         "n_epochs": 5,
#         "n_groups_per_batch": 1,
#         "unlabeled_n_groups_per_batch": 1,
#         "algo_log_metric": "accuracy",
#         "max_token_length": 300,
#         "irm_lambda": 1.0,
#         "coral_penalty_weight": 10.0,
#         "dann_penalty_weight": 1.0,
#         "dann_featurizer_lr": 1e-6,
#         "dann_classifier_lr": 1e-5,
#         "dann_discriminator_lr": 1e-5,
#         "loader_kwargs": {
#             "num_workers": 1,
#             "pin_memory": True,
#         },
#         "unlabeled_loader_kwargs": {
#             "num_workers": 1,
#             "pin_memory": True,
#         },
#         "process_outputs_function": "multiclass_logits_to_pred",
#         "process_pseudolabels_function": "pseudolabel_multiclass_logits",
#     },
#     "domainnet": {
#         "split_scheme": "official",
#         "dataset_kwargs": {
#             "source_domain": "real",
#             "target_domain": "sketch",
#             "use_sentry": False,
#         },
#         "model": "resnet50",
#         "model_kwargs": {"pretrained": True},
#         "transform": "image_resize",
#         "resize_scale": 256.0 / 224.0,
#         "target_resolution": (224, 224),
#         "loss_function": "cross_entropy",
#         "groupby_fields": [
#             "category",
#         ],
#         "val_metric": "acc_avg",
#         "val_metric_decreasing": False,
#         "batch_size": 96,
#         "unlabeled_batch_size": 224,
#         "optimizer": "SGD",
#         "optimizer_kwargs": {
#             "momentum": 0.9,
#         },
#         "lr": 0.0007035737028722148,
#         "weight_decay": 1e-4,
#         "n_epochs": 25,
#         "n_groups_per_batch": 4,
#         "unlabeled_n_groups_per_batch": 4,
#         "irm_lambda": 1.0,
#         "coral_penalty_weight": 1.0,
#         "dann_penalty_weight": 1.0,
#         "dann_featurizer_lr": 0.001,
#         "dann_classifier_lr": 0.01,
#         "dann_discriminator_lr": 0.01,
#         "algo_log_metric": "accuracy",
#         "process_outputs_function": "multiclass_logits_to_pred",
#         "process_pseudolabels_function": "pseudolabel_multiclass_logits",
#         "loader_kwargs": {
#             "num_workers": 2,
#             "pin_memory": True,
#         },
#     },
#     "encode": {
#         "split_scheme": "official",
#         "model": "unet-seq",
#         "model_kwargs": {"n_channels_in": 5},
#         "loader_kwargs": {
#             "num_workers": 1
#         },  # pybigwig seems to have trouble with multiprocessing
#         "transform": None,
#         "loss_function": "multitask_bce",
#         "groupby_fields": ["celltype"],
#         "val_metric": "avgprec-macro_all",
#         "val_metric_decreasing": False,
#         "optimizer": "Adam",
#         "scheduler": "MultiStepLR",
#         "scheduler_kwargs": {"milestones": [3, 6], "gamma": 0.1},
#         "batch_size": 128,
#         "lr": 1e-3,
#         "weight_decay": 1e-4,
#         "n_epochs": 12,
#         "n_groups_per_batch": 4,
#         "algo_log_metric": "multitask_binary_accuracy",
#         "irm_lambda": 100.0,
#         "coral_penalty_weight": 0.1,
#     },
#     "fmow": {
#         "split_scheme": "official",
#         "dataset_kwargs": {"seed": 111, "use_ood_val": True},
#         "model": "densenet121",
#         "model_kwargs": {"pretrained": True},
#         "transform": "image_base",
#         "loss_function": "cross_entropy",
#         "groupby_fields": [
#             "year",
#         ],
#         "val_metric": "acc_worst_region",
#         "val_metric_decreasing": False,
#         "optimizer": "Adam",
#         "scheduler": "StepLR",
#         "scheduler_kwargs": {"gamma": 0.96},
#         "batch_size": 32,
#         "unlabeled_batch_size": 32,
#         "lr": 0.0001,
#         "weight_decay": 0.0,
#         "n_epochs": 60,
#         "n_groups_per_batch": 8,
#         "unlabeled_n_groups_per_batch": 8,
#         "irm_lambda": 1.0,
#         "coral_penalty_weight": 0.1,
#         "dann_penalty_weight": 1.0,
#         "dann_featurizer_lr": 0.00001,
#         "dann_classifier_lr": 0.0001,
#         "dann_discriminator_lr": 0.0001,
#         "algo_log_metric": "accuracy",
#         "process_outputs_function": "multiclass_logits_to_pred",
#         "process_pseudolabels_function": "pseudolabel_multiclass_logits",
#     },
#     "iwildcam": {
#         "loss_function": "cross_entropy",
#         "val_metric": "F1-macro_all",
#         "model_kwargs": {"pretrained": True},
#         "transform": "image_base",
#         "target_resolution": (448, 448),
#         "val_metric_decreasing": False,
#         "algo_log_metric": "accuracy",
#         "model": "resnet50",
#         "lr": 3e-5,
#         "weight_decay": 0.0,
#         "batch_size": 16,
#         "unlabeled_batch_size": 16,
#         "n_epochs": 12,
#         "optimizer": "Adam",
#         "split_scheme": "official",
#         "scheduler": None,
#         "groupby_fields": [
#             "location",
#         ],
#         "n_groups_per_batch": 2,
#         "unlabeled_n_groups_per_batch": 2,
#         "irm_lambda": 1.0,
#         "coral_penalty_weight": 10.0,
#         "dann_penalty_weight": 0.1,
#         "dann_featurizer_lr": 3e-6,
#         "dann_classifier_lr": 3e-5,
#         "dann_discriminator_lr": 3e-5,
#         "no_group_logging": True,
#         "process_outputs_function": "multiclass_logits_to_pred",
#         "process_pseudolabels_function": "pseudolabel_multiclass_logits",
#     },
#     "ogb-molpcba": {
#         "split_scheme": "official",
#         "model": "gin-virtual",
#         "model_kwargs": {"dropout": 0.5},  # include pretrained
#         "loss_function": "multitask_bce",
#         "groupby_fields": [
#             "scaffold",
#         ],
#         "val_metric": "ap",
#         "val_metric_decreasing": False,
#         "optimizer": "Adam",
#         "batch_size": 32,
#         "unlabeled_batch_size": 32,
#         "lr": 1e-3,
#         "weight_decay": 0.0,
#         "n_epochs": 100,
#         "n_groups_per_batch": 4,
#         "unlabeled_n_groups_per_batch": 4,
#         "irm_lambda": 1.0,
#         "coral_penalty_weight": 0.1,
#         "dann_penalty_weight": 0.1,
#         "dann_featurizer_lr": 1e-3,
#         "dann_classifier_lr": 1e-2,
#         "dann_discriminator_lr": 1e-2,
#         "noisystudent_add_dropout": False,
#         "no_group_logging": True,
#         "algo_log_metric": "multitask_binary_accuracy",
#         "process_outputs_function": None,
#         "process_pseudolabels_function": "pseudolabel_binary_logits",
#         "loader_kwargs": {
#             "num_workers": 1,
#             "pin_memory": True,
#         },
#     },
#     "py150": {
#         "split_scheme": "official",
#         "model": "code-gpt-py",
#         "loss_function": "lm_cross_entropy",
#         "val_metric": "acc",
#         "val_metric_decreasing": False,
#         "optimizer": "AdamW",
#         "optimizer_kwargs": {"eps": 1e-8},
#         "lr": 8e-5,
#         "weight_decay": 0.0,
#         "n_epochs": 3,
#         "batch_size": 6,
#         "groupby_fields": [
#             "repo",
#         ],
#         "n_groups_per_batch": 2,
#         "irm_lambda": 1.0,
#         "coral_penalty_weight": 1.0,
#         "no_group_logging": True,
#         "algo_log_metric": "multitask_accuracy",
#         "process_outputs_function": "multiclass_logits_to_pred",
#     },
#     "poverty": {
#         "split_scheme": "official",
#         "dataset_kwargs": {"no_nl": False, "fold": "A", "use_ood_val": True},
#         "model": "resnet18_ms",
#         "model_kwargs": {"num_channels": 8},
#         "transform": "poverty",
#         "loss_function": "mse",
#         "groupby_fields": [
#             "country",
#         ],
#         "val_metric": "r_wg",
#         "val_metric_decreasing": False,
#         "algo_log_metric": "mse",
#         "optimizer": "Adam",
#         "scheduler": "StepLR",
#         "scheduler_kwargs": {"gamma": 0.96},
#         "batch_size": 64,
#         "unlabeled_batch_size": 64,
#         "lr": 0.001,
#         "weight_decay": 0.0,
#         "n_epochs": 200,
#         "n_groups_per_batch": 8,
#         "unlabeled_n_groups_per_batch": 4,
#         "irm_lambda": 1.0,
#         "coral_penalty_weight": 0.1,
#         "dann_penalty_weight": 0.1,
#         "dann_featurizer_lr": 0.0001,
#         "dann_classifier_lr": 0.001,
#         "dann_discriminator_lr": 0.001,
#         "process_outputs_function": None,
#         "process_pseudolabels_function": "pseudolabel_identity",
#     },
#     "waterbirds": {
#         "split_scheme": "official",
#         "model": "resnet50",
#         "transform": "image_resize_and_center_crop",
#         "resize_scale": 256.0 / 224.0,
#         "model_kwargs": {"pretrained": True},
#         "loss_function": "cross_entropy",
#         "groupby_fields": ["background", "y"],
#         "val_metric": "acc_wg",
#         "val_metric_decreasing": False,
#         "algo_log_metric": "accuracy",
#         "optimizer": "SGD",
#         "optimizer_kwargs": {"momentum": 0.9},
#         "scheduler": None,
#         "batch_size": 128,
#         "lr": 1e-5,
#         "weight_decay": 1.0,
#         "n_epochs": 300,
#         "process_outputs_function": "multiclass_logits_to_pred",
#     },
#     "yelp": {
#         "split_scheme": "official",
#         "model": "bert-base-uncased",
#         "transform": "bert",
#         "max_token_length": 512,
#         "loss_function": "cross_entropy",
#         "algo_log_metric": "accuracy",
#         "batch_size": 8,
#         "lr": 2e-6,
#         "weight_decay": 0.01,
#         "n_epochs": 3,
#         "n_groups_per_batch": 2,
#         "process_outputs_function": "multiclass_logits_to_pred",
#     },
#     "sqf": {
#         "split_scheme": "all_race",
#         "model": "logistic_regression",
#         "transform": None,
#         "model_kwargs": {"in_features": 104},
#         "loss_function": "cross_entropy",
#         "groupby_fields": ["y"],
#         "val_metric": "precision_at_global_recall_all",
#         "val_metric_decreasing": False,
#         "algo_log_metric": "accuracy",
#         "optimizer": "Adam",
#         "optimizer_kwargs": {},
#         "scheduler": None,
#         "batch_size": 4,
#         "lr": 5e-5,
#         "weight_decay": 0,
#         "n_epochs": 4,
#         "process_outputs_function": None,
#     },
#     "rxrx1": {
#         "split_scheme": "official",
#         "model": "resnet50",
#         "model_kwargs": {"pretrained": True},
#         "transform": "rxrx1",
#         "target_resolution": (256, 256),
#         "loss_function": "cross_entropy",
#         "groupby_fields": ["experiment"],
#         "val_metric": "acc_avg",
#         "val_metric_decreasing": False,
#         "algo_log_metric": "accuracy",
#         "optimizer": "Adam",
#         "optimizer_kwargs": {},
#         "scheduler": "cosine_schedule_with_warmup",
#         "scheduler_kwargs": {"num_warmup_steps": 5415},
#         "batch_size": 72,
#         "lr": 1e-3,
#         "weight_decay": 1e-5,
#         "n_groups_per_batch": 9,
#         "coral_penalty_weight": 0.1,
#         "irm_lambda": 1.0,
#         "n_epochs": 90,
#         "process_outputs_function": "multiclass_logits_to_pred",
#     },
#     "globalwheat": {
#         "split_scheme": "official",
#         "model": "fasterrcnn",
#         "transform": "image_base",
#         "model_kwargs": {"n_classes": 1, "pretrained": True},
#         "loss_function": "fasterrcnn_criterion",
#         "groupby_fields": ["session"],
#         "val_metric": "detection_acc_avg_dom",
#         "val_metric_decreasing": False,
#         "algo_log_metric": None,  # TODO
#         "optimizer": "Adam",
#         "optimizer_kwargs": {},
#         "scheduler": None,
#         "batch_size": 4,
#         "unlabeled_batch_size": 4,
#         "lr": 1e-5,
#         "weight_decay": 1e-3,
#         "n_epochs": 12,
#         "noisystudent_add_dropout": False,
#         "self_training_threshold": 0.5,
#         "loader_kwargs": {
#             "num_workers": 1,
#             "pin_memory": True,
#         },
#         "process_outputs_function": None,
#         "process_pseudolabels_function": "pseudolabel_detection_discard_empty",
#     },
# }


# def make_split_name(split, unlabeled):
#     assert split in SPLITS
#     if unlabeled:
#         suffix = "_unlabeled"
#     else:
#         suffix = ''
#     return f"{split}{suffix}"


def get_wilds_dataloaders(
    dataset_type,
    train_batch_size,
    test_batch_size,
    train_transform=None,
    test_transform=None,
    root_dir=None,
    fraction=1,
    unlabeled=False,
    grouping_kwargs={},
    keep_metadata=True,
    use_waterbirds_divdis=False,
    logger=None
):

    if root_dir is None and not use_waterbirds_divdis:
        root_dir = make_default_data_path()

    dataset_kwargs = {}
    if root_dir is not None:
        dataset_kwargs["root_dir"] = root_dir

    train_split = make_split_name("train", unlabeled)
    val_split = make_split_name("val", unlabeled)
    test_split = make_split_name("test", unlabeled)

    if use_waterbirds_divdis:

        assert root_dir is None
        assert unlabeled is False
        assert grouping_kwargs == {}
        assert train_transform is None
        assert test_transform is None
        assert dataset_type == "waterbirds"

        train_dataloader, val_dataloader, test_dataloader \
            = make_divdis_dataloaders(
                train_batch_size,
                test_batch_size,
                fraction,
                keep_metadata,
                logger
            )

    else:

        # Load the full dataset, and download it if necessary
        dataset = get_dataset(
            dataset=dataset_type,
            download=True,
            **dataset_kwargs
        )

        if unlabeled:
            unlabeled_dataset = get_dataset(
                dataset=dataset_type,
                download=True,
                unlabeled=unlabeled,
                **dataset_kwargs
            )
        else:
            unlabeled_dataset = None

        train_dataloader = None
        val_dataloader = None
        test_dataloader = None

        if train_batch_size > 0:
            train_dataloader = make_dataloader(
                dataset,
                unlabeled_dataset,
                train_batch_size,
                train_split,
                train_transform,
                fraction=fraction,
                grouping_kwargs=grouping_kwargs,
                keep_metadata=keep_metadata
            )

        if test_batch_size > 0:

            val_dataloader = make_dataloader(
                dataset,
                unlabeled_dataset,
                test_batch_size,
                val_split,
                test_transform,
                fraction=fraction,
                grouping_kwargs=grouping_kwargs,
                keep_metadata=keep_metadata,
                train=False
            )

            test_dataloader = make_dataloader(
                dataset,
                unlabeled_dataset,
                test_batch_size,
                test_split,
                test_transform,
                fraction=fraction,
                grouping_kwargs=grouping_kwargs,
                keep_metadata=keep_metadata,
                train=False
            )

    return (
        train_dataloader,
        {
            val_split: val_dataloader,
            test_split: test_dataloader
        }
    )


# def make_dataloader(
#     dataset,
#     unlabeled_dataset,
#     batch_size,
#     subset_name,
#     transform,
#     loader_type="standard",
#     fraction=1.0,
#     grouping_kwargs={},
#     keep_metadata=True,
#     train=True
# ):

#     config_template = DATASET_DEFAULTS.get(dataset.dataset_name, {})

#     grouping_kwargs.setdefault("groupby_fields")
#     groupby_fields = grouping_kwargs.pop("groupby_fields")

#     datasets_for_grouper = [dataset]

#     if unlabeled_dataset is not None:
#         datasets_for_grouper += [unlabeled_dataset]

#     grouper = CombinatorialGrouper(
#         dataset=datasets_for_grouper,
#         groupby_fields=groupby_fields
#     )
#     grouper.n_groups_per_batch \
#         = grouping_kwargs.setdefault(
#             "n_groups_per_batch",
#             config_template.get("n_groups_per_batch")
#         )
#     grouper.distinct_groups \
#         = grouping_kwargs.setdefault("distinct_groups")
#     grouping_kwargs.setdefault("uniform_over_groups")

#     if transform is None:
#         default_res = config_template.get("target_resolution", WATERBIRDS_RES)
#         transform = transforms.Compose(
#             [transforms.Resize(default_res), transforms.ToTensor()]
#         )
#     data = datasets_for_grouper[-1].get_subset(
#         subset_name,
#         transform=transform,
#         frac=fraction
#     )

#     if train:
#         dataloader = get_train_loader(
#             loader_type,
#             data,
#             batch_size=batch_size,
#             **grouping_kwargs
#         )
#     else:
#         dataloader = get_eval_loader(
#             loader_type,
#             data,
#             batch_size=batch_size
#         )
#     dataloader.grouper = grouper

#     dataloader = wrap_if_needed(dataloader, keep_metadata)

#     return dataloader


# def wrap_if_needed(dataloader, keep_metadata, k=1):

#     if dataloader is None:
#         return dataloader

#     if keep_metadata:
#         dataloader.has_metadata = True
#         return dataloader
#     else:
#         # TODO(Alex | 15.05.2024): drop last k instead of 1 in wrapper itself
#         for _ in range(k):
#             dataloader.has_metadata = False
#             dataloader = wrap_dataloader(
#                 dataloader,
#                 wrapper=DropLastIteratorWrapper
#             )
#         return dataloader


def metadata_to_group(metadata):
    if len(metadata.shape) == 1:
        return metadata
    else:
        return (metadata * torch.tensor([1, 2, 0])).sum(-1)


# #### From DivDis
# # based on: https://github.com/yoonholee/DivDis/blob/b9de1a637949594054240254f667063788ee1573/subpopulation/data/confounder_utils.py#L33

# # TODO(Alex | 15.05.2024): Move to separate file
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.sampler import WeightedRandomSampler
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import Dataset
# import pandas as pd
# # from data.folds import Subset


# def download_waterbirds_divdis(root_dir, logger):
#     data_path = os.path.join(
#         root_dir,
#         "data",
#     )
#     file_path = os.path.join(
#         data_path,
#         "waterbird_complete95_forest2water2"
#     )
#     archive_path = file_path + ".tar.gz"
#     if not os.path.exists(file_path):
#         if not os.path.exists(archive_path):
#             log_or_print(
#                 f"Downloading waterbirds data \nfrom {WATERBIRDS_DIVDIS_LINK} \nto {data_path}",
#                 logger,
#                 auto_newline=True
#             )
#             os.makedirs(data_path, exist_ok=True)
#             runcmd(
#                 f"cd {data_path} && wget {WATERBIRDS_DIVDIS_LINK}",
#                 verbose=True,
#                 logger=logger
#             )
#         log_or_print(
#             f"Extracting waterbirds data \nfrom {archive_path} \nto {file_path}",
#             logger,
#             auto_newline=True
#         )
#         runcmd(
#             f"cd {data_path} && tar -xf {archive_path}",
#             verbose=True,
#             logger=logger
#         )
#         log_or_print(
#             f"Removing archive \nfrom {archive_path}",
#             logger,
#             auto_newline=True
#         )
#         runcmd(
#             f"cd {data_path} && rm {archive_path}",
#             verbose=True,
#             logger=logger
#         )


# def make_divdis_dataloaders(
#     train_batch_size,
#     test_batch_size,
#     fraction,
#     keep_metadata,
#     logger
# ):

#     args = make_dummy_object()
#     args.root_dir = os.path.join(get_project_root_path(), "submodules", "div-dis")
#     download_waterbirds_divdis(args.root_dir, logger)

#     args.target_name = 'waterbird_complete95'
#     args.confounder_names = ['forest2water2']
#     args.model = 'resnet50'
#     args.augment_data = False
#     args.group_by_label = False
#     args.fraction = fraction

#     train_data, val_data, test_data = prepare_confounder_data(
#         args,
#         train=True,
#         return_full_dataset=False
#     )

#     loader_kwargs = {
#         "num_workers": 4,
#         "pin_memory": True
#     }

#     if train_batch_size > 0:
#         loader_kwargs["batch_size"] = train_batch_size
#         train_dataloader = construct_loader(
#             train_data, train=True, reweight_groups=None, loader_kwargs=loader_kwargs
#         )
#     else:
#         train_dataloader = None

#     if test_batch_size > 0:
#         loader_kwargs["batch_size"] = test_batch_size
#         val_dataloader = construct_loader(
#             val_data, train=False, reweight_groups=None, loader_kwargs=loader_kwargs
#         )

#         test_dataloader = construct_loader(
#             test_data, train=False, reweight_groups=None, loader_kwargs=loader_kwargs
#         )
#     else:
#         val_dataloader, test_dataloader = None, None

#     return (
#         wrap_if_needed(train_dataloader, keep_metadata, k=2),
#         wrap_if_needed(val_dataloader, keep_metadata, k=2),
#         wrap_if_needed(test_dataloader, keep_metadata, k=2)
#     )


# def construct_loader(data, train, reweight_groups, loader_kwargs):
#     if data is not None:
#         return data.get_loader(
#             train=train, reweight_groups=reweight_groups, **loader_kwargs
#         )


# def prepare_confounder_data(args, train, return_full_dataset=False):
#     # full_dataset = confounder_settings[args.dataset]["constructor"](
#     full_dataset = CUBDataset(
#         args=args,
#         root_dir=args.root_dir,
#         target_name=args.target_name,
#         confounder_names=args.confounder_names,
#         model_type=args.model,
#         augment_data=args.augment_data,
#     )
#     if return_full_dataset:
#         return DRODataset(
#             full_dataset,
#             process_item_fn=None,
#             n_groups=full_dataset.n_groups,
#             n_classes=full_dataset.n_classes,
#             group_str_fn=full_dataset.group_str,
#         )
#     if train:
#         splits = ["train", "val", "test"]
#     else:
#         splits = ["test"]
#     subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
#     dro_subsets = [
#         DRODataset(
#             subsets[split],
#             process_item_fn=None,
#             n_groups=full_dataset.n_groups,
#             n_classes=full_dataset.n_classes,
#             group_str_fn=full_dataset.group_str,
#         )
#         for split in splits
#     ]
#     return dro_subsets


# class DRODataset(Dataset):
#     def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn):
#         self.dataset = dataset
#         self.process_item = process_item_fn
#         self.n_groups = n_groups
#         self.n_classes = n_classes
#         self.group_str = group_str_fn
#         # group_array = []
#         # y_array = []
#         # for batch in self:
#         #     group_array.append(batch[2])
#         #     y_array.append(batch[1])
#         # self._group_array = torch.LongTensor(group_array)
#         # self._y_array = torch.LongTensor(y_array)

#         # self._group_array = torch.LongTensor(dataset.dataset.group_array[dataset.indices])
#         # self._y_array = torch.LongTensor(dataset.dataset.y_array[dataset.indices])

#         self._group_array = torch.LongTensor(dataset.get_group_array())
#         self._y_array = torch.Tensor(dataset.get_label_array())
#         self._group_counts = (
#             (torch.arange(self.n_groups).unsqueeze(1) == self._group_array)
#             .sum(1)
#             .float()
#         )
#         self._group_counts = self._group_counts[np.where(self._group_counts > 0)]
#         self._y_counts = (
#             (torch.arange(self.n_classes).unsqueeze(1) == self._y_array).sum(1).float()
#         )
#         self.group_indices = {
#             loc.item(): torch.nonzero(self._group_array == loc).squeeze(-1)
#             for loc in self._group_array.unique()
#         }
#         self.distinct_groups = np.unique(self._group_array)

#         assert len(self._group_array) == len(self.dataset)

#     def get_sample(self, g, idx, cross=False):
#         g = g.item()
#         if cross:
#             g = np.random.choice(np.setdiff1d(self.distinct_groups, [g]))
#         new_idx = np.random.choice(self.group_indices[g].numpy())
#         # while new_idx == idx:
#         #     new_idx = np.random.choice(self.group_indices[g].numpy())
#         # if idx >= len(self.dataset):
#         #     pdb.set_trace()
#         return self.dataset[new_idx]

#     def __getitem__(self, idx):

#         if self.process_item is None:
#             return self.dataset[idx]
#         else:
#             return self.process_item(self.dataset[idx])

#     def __len__(self):
#         return len(self.dataset)

#     def group_counts(self):
#         return self._group_counts

#     def class_counts(self):
#         return self._y_counts

#     def input_size(self):
#         for sample in self:
#             x = sample[0]
#             return x.size()

#     def get_group_array(self):
#         if self.process_item is None:
#             return self.dataset.get_group_array()
#         else:
#             raise NotImplementedError

#     def get_label_array(self):
#         if self.process_item is None:
#             return self.dataset.get_label_array()
#         else:
#             raise NotImplementedError

#     def get_loader(self, train, reweight_groups, **kwargs):
#         if not train:  # Validation or testing
#             assert reweight_groups is None
#             shuffle = False
#             sampler = None
#         elif not reweight_groups:  # Training but not reweighting
#             shuffle = True
#             sampler = None
#         else:  # Training and reweighting
#             # When the --robust flag is not set, reweighting changes the loss function
#             # from the normal ERM (average loss over each training example)
#             # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
#             # When the --robust flag is set, reweighting does not change the loss function
#             # since the minibatch is only used for mean gradient estimation for each group separately
#             group_weights = len(self) / self._group_counts
#             weights = group_weights[self._group_array]

#             assert not np.isnan(weights).any()

#             # Replacement needs to be set to True, otherwise we'll run out of minority samples
#             sampler = WeightedRandomSampler(weights, len(self), replacement=True)
#             shuffle = False

#         loader = DataLoader(self, shuffle=shuffle, sampler=sampler, **kwargs)
#         return loader


# def get_transform_cub(model_type, train, augment_data):
#     scale = 256.0 / 224.0
#     # target_resolution = model_attributes[model_type]["target_resolution"]
#     target_resolution = (224, 224)
#     assert target_resolution is not None

#     if (not train) or (not augment_data):
#         # Resizes the image to a slightly larger square then crops the center.
#         transform = transforms.Compose(
#             [
#                 transforms.Resize(
#                     (
#                         int(target_resolution[0] * scale),
#                         int(target_resolution[1] * scale),
#                     )
#                 ),
#                 transforms.CenterCrop(target_resolution),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         )
#     else:
#         transform = transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(
#                     target_resolution,
#                     scale=(0.7, 1.0),
#                     ratio=(0.75, 1.3333333333333333),
#                     interpolation=2,
#                 ),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         )
#     return transform


# class Subset(torch.utils.data.Dataset):
#     """
#     Subsets a dataset while preserving original indexing.

#     NOTE: torch.utils.dataset.Subset loses original indexing.
#     """

#     def __init__(self, dataset, indices):
#         self.dataset = dataset
#         self.indices = indices

#         self.group_array = self.get_group_array(re_evaluate=True)
#         self.label_array = self.get_label_array(re_evaluate=True)

#     def __getitem__(self, idx):
#         return self.dataset[self.indices[idx]]

#     def __len__(self):
#         return len(self.indices)

#     def get_group_array(self, re_evaluate=True):
#         """Return an array [g_x1, g_x2, ...]"""
#         # setting re_evaluate=False helps us over-write the group array if necessary (2-group DRO)
#         if re_evaluate:
#             group_array = self.dataset.get_group_array()[self.indices]
#             assert len(group_array) == len(
#                 self.indices
#             ), f"length of self.group_array:{len(group_array)}, length of indices:{len(self.indices)}"
#             assert len(self.indices) == len(self)
#             assert len(group_array) == len(self)
#             return group_array
#         else:
#             return self.group_array

#     def get_label_array(self, re_evaluate=True):
#         if re_evaluate:
#             label_array = self.dataset.get_label_array()[self.indices]
#             assert len(label_array) == len(self)
#             return label_array
#         else:
#             return self.label_array


# class ConfounderDataset(Dataset):
#     def __init__(
#         self,
#         root_dir,
#         target_name,
#         confounder_names,
#         model_type=None,
#         augment_data=None,
#     ):
#         raise NotImplementedError

#     def __len__(self):
#         return len(self.group_array)

#     def __getitem__(self, idx):
#         g = self.group_array[idx]
#         y = self.y_array[idx]

#         # if model_attributes[self.model_type]['feature_type']=='precomputed':
#         if self.precomputed:
#             x = self.features_mat[idx]
#             if not self.pretransformed:
#                 if self.split_array[idx] == 0:
#                     x = self.train_transform(x)
#                 else:
#                     x = self.eval_transform(x)

#             assert not isinstance(x, list)
#         else:
#             if not hasattr(self, "mix_array") or not self.mix_array[idx]:
#                 x = self.get_image(idx)
#             else:
#                 idx_1, idx_2 = self.mix_idx_array[idx]
#                 x1, x2 = self.get_image(idx_1), self.get_image(idx_2)

#                 l = self.mix_weight_array[idx]

#                 x = l * x1 + (1 - l) * x2

#         if self.mix_up:
#             y_onehot = self.y_array_onehot[idx]

#             try:
#                 true_g = self.domains[idx]
#             except:
#                 true_g = None

#             if true_g is None:
#                 return x, y, g, y_onehot, idx

#             else:
#                 return x, y, true_g, y_onehot, idx

#         else:
#             return x, y, g, idx

#     def refine_dataset(self):
#         for name, split_id in self.split_dict.items():
#             idxes = np.where(self.split_array == split_id)
#             group_counts = (
#                 (
#                     torch.arange(self.n_groups).unsqueeze(1)
#                     == torch.tensor(self.group_array[idxes])
#                 )
#                 .sum(1)
#                 .float()
#             )
#             unique_group_id = torch.where(group_counts > 0)[0]
#             # unique_group_id, counts = np.unique(self.group_array[idxes], return_counts=True)
#             # unique_group_id = unique_group_id[np.where(counts) > 0]
#             group_dict = {
#                 id: new_id for new_id, id in enumerate(unique_group_id.tolist())
#             }
#             self.group_array[idxes] = np.array(
#                 [group_dict[id] for id in self.group_array[idxes]]
#             )

#     def get_image(self, idx):
#         # img_filename = os.path.join(
#         #     self.data_dir,
#         #     self.filename_array[idx])
#         img_filename = self.filename_array[idx]
#         img = Image.open(img_filename)
#         if self.RGB:
#             img = img.convert("RGB")
#         # Figure out split and transform accordingly
#         if self.split_array[idx] == self.split_dict["train"] and self.train_transform:
#             img = self.train_transform(img)
#         elif (
#             self.split_array[idx] in [self.split_dict["val"], self.split_dict["test"]]
#             and self.eval_transform
#         ):
#             img = self.eval_transform(img)
#         # Flatten if needed
#         # if model_attributes[self.model_type]["flatten"]:
#         if False:
#             assert img.dim() == 3
#             img = img.view(-1)
#         return img

#     def get_splits(self, splits, train_frac=1.0):
#         subsets = {}
#         for split in splits:
#             # assert split in ('train','val','test'), split+' is not a valid split'
#             mask = self.split_array == self.split_dict[split]
#             num_split = np.sum(mask)
#             indices = np.where(mask)[0]
#             if train_frac < 1 and split == "train":
#                 num_to_retain = int(np.round(float(len(indices)) * train_frac))
#                 indices = np.sort(np.random.permutation(indices)[:num_to_retain])
#             subsets[split] = Subset(self, indices)
#         return subsets

#     def group_str(self, group_idx):
#         y = group_idx // (self.n_groups / self.n_classes)
#         c = group_idx % (self.n_groups // self.n_classes)

#         group_name = f"{self.target_name} = {int(y)}"
#         bin_str = format(int(c), f"0{self.n_confounders}b")[::-1]
#         for attr_idx, attr_name in enumerate(self.confounder_names):
#             group_name += f", {attr_name} = {bin_str[attr_idx]}"
#         return group_name


# class CUBDataset(ConfounderDataset):
#     """
#     CUB dataset (already cropped and centered).
#     Note: metadata_df is one-indexed.
#     """

#     def __init__(
#         self,
#         args,
#         root_dir,
#         target_name,
#         confounder_names,
#         augment_data=False,
#         model_type=None,
#         mix_up=False,
#         group_id=None,
#         dataset=None,
#     ):
#         self.args = args
#         self.mix_up = mix_up
#         self.root_dir = root_dir
#         self.target_name = target_name
#         self.confounder_names = confounder_names
#         self.model_type = model_type
#         self.augment_data = augment_data

#         self.data_dir = os.path.join(
#             self.root_dir, "data", "_".join([self.target_name] + self.confounder_names)
#         )

#         if not os.path.exists(self.data_dir) and not os.path.exists(
#             os.path.join(root_dir, "features", "cub.npy")
#         ):
#             raise ValueError(
#                 f"{self.data_dir} does not exist yet. Please generate the dataset first."
#             )

#         # Read in metadata
#         self.metadata_df = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))

#         # Get the y values
#         self.y_array = self.metadata_df["y"].values
#         self.n_classes = 2

#         # We only support one confounder for CUB for now
#         self.confounder_array = self.metadata_df["place"].values
#         self.n_confounders = 1

#         # Extract filenames and splits
#         self.filename_array = self.metadata_df["img_filename"].values
#         self.split_array = self.metadata_df["split"].values
#         self.split_dict = {"train": 0, "val": 1, "test": 2}

#         # Map to groups
#         self.n_groups = pow(2, 2)
#         self.group_array = (
#             self.y_array * (self.n_groups / 2) + self.confounder_array
#         ).astype("int")

#         if args.group_by_label:
#             idxes = np.where(self.split_array == self.split_dict["train"])[0]
#             self.group_array[idxes] = self.y_array[idxes]

#         # Set transform
#         # if model_attributes[self.model_type]['feature_type']=='precomputed':
#         self.precomputed = True
#         self.pretransformed = True
#         if os.path.exists(os.path.join(root_dir, "features", "cub.npy")):
#             self.features_mat = torch.from_numpy(
#                 np.load(
#                     os.path.join(root_dir, "features", "cub.npy"), allow_pickle=True
#                 )
#             ).float()
#             self.train_transform = None
#             self.eval_transform = None
#         else:
#             self.features_mat = []
#             self.train_transform = get_transform_cub(
#                 self.model_type, train=True, augment_data=augment_data
#             )
#             self.eval_transform = get_transform_cub(
#                 self.model_type, train=False, augment_data=augment_data
#             )

#             for idx in tqdm(range(len(self.y_array))):
#                 img_filename = os.path.join(self.data_dir, self.filename_array[idx])
#                 img = Image.open(img_filename).convert("RGB")
#                 # Figure out split and transform accordingly
#                 if (
#                     self.split_array[idx] == self.split_dict["train"]
#                     and self.train_transform
#                 ):
#                     img = self.train_transform(img)
#                 elif (
#                     self.split_array[idx]
#                     in [self.split_dict["val"], self.split_dict["test"]]
#                     and self.eval_transform
#                 ):
#                     img = self.eval_transform(img)
#                 # Flatten if needed
#                 # if model_attributes[self.model_type]["flatten"]:
#                 if False:
#                     assert img.dim() == 3
#                     img = img.view(-1)
#                 x = img
#                 self.features_mat.append(x)

#             self.features_mat = torch.cat(
#                 [x.unsqueeze(0) for x in self.features_mat], dim=0
#             )
#             os.makedirs(os.path.join(root_dir, "features"))
#             np.save(
#                 os.path.join(root_dir, "features", "cub.npy"), self.features_mat.numpy()
#             )

#         self.RGB = True

#         self.y_array_onehot = torch.zeros(len(self.y_array), self.n_classes)
#         self.y_array_onehot = self.y_array_onehot.scatter_(
#             1, torch.tensor(self.y_array).unsqueeze(1), 1
#         ).numpy()
#         self.original_train_length = len(self.features_mat)

#         if group_id is not None:
#             idxes = np.where(self.group_array == group_id)
#             self.select_samples(idxes)

#     def select_samples(self, idxes):
#         self.y_array = self.y_array[idxes]
#         self.group_array = self.group_array[idxes]
#         self.split_array = self.split_array[idxes]
#         self.features_mat = self.features_mat[idxes]
#         self.y_array_onehot = self.y_array_onehot[idxes]

#     def get_group_array(self):
#         return self.group_array

#     def get_label_array(self):
#         return self.y_array
