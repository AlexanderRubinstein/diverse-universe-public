# import os
import sys
# import torch
from stuned.utility.utils import (
    NAME_NUMBER_SEPARATOR,
    get_project_root_path,
    get_with_assert,
    raise_unknown,
    get_hash,
    parse_name_and_number
)
from stuned.local_datasets.utils import (
    make_default_cache_path,
    chain_dataloaders,
    randomly_subsampled_dataloader,
    # make_or_load_from_cache
)
from stuned.local_datasets.transforms import (
    make_transforms,
    make_default_test_transforms_imagenet
)
# from stuned.local_datasets.features_labeller import (
#     make_features_labeller,
#     load_features_labeller
# )
from stuned.local_datasets.imagenet1k import (
    get_imagenet_dataloaders
)


# local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(
    0,
    # os.path.join(
    #     os.path.dirname(os.path.abspath('')), "src"
    # )
    get_project_root_path()
)
# from utility.logger import make_logger
# from utility.utils import (
#     NAME_NUMBER_SEPARATOR,
#     raise_unknown,
#     parse_name_and_number,
#     get_hash,
#     get_with_assert
# )
# from local_datasets.features_labeller import (
#     make_features_labeller,
#     load_features_labeller
# )
# from local_datasets.tiny_imagenet import (
#     TINY_IMAGENET,
#     TINY_IMAGENET_TEST,
#     get_tiny_imagenet_dataloaders,
#     get_tiny_imagenet_test_projector_dataset
# )
# from local_datasets.dsprites import (
#     get_dsprites_unlabeled_data
# )
# from local_datasets.synthetic_pos import get_synthetic_pos_dataloaders
from diverse_universe.local_datasets.wilds import (
    get_wilds_dataloaders
)
from diverse_universe.local_datasets.imagenet_c import (
    get_imagenet_c_dataloaders
)
from diverse_universe.local_datasets.openimages import (
    get_openimages_dataloader
)
# from local_datasets.utils import (
#     TRAIN_KEY,
#     randomly_subsampled_dataloader,
#     make_or_load_from_cache,
#     chain_dataloaders,
#     make_default_cache_path
# )
# from local_datasets.imagenet1k import (
#     get_imagenet_dataloaders
# )
from diverse_universe.local_datasets.model_vs_human import (
    get_modelvshuman_dataloaders
)
# from local_datasets.from_folder import (
#     get_dataloaders_from_folder
# )
# from local_datasets.mnist_cifar import (
#     get_mnist_cifar_dataloaders
# )
# from local_datasets.transforms import (
#     make_transforms
# )
# from local_datasets.utk_face import (
#     make_utk_face_base_data
# )
from diverse_universe.local_datasets.from_h5 import (
    FROM_H5,
    MAX_CHUNK_SIZE,
    get_h5_dataloaders
)
from diverse_universe.local_datasets.easy_robust import (
    # get_easy_robust_dataloaders
    get_imagenet_arv2_dataloader
)
# from local_models.utils import (
#     separate_classifier_and_featurizer
# )
# from local_models.common import (
#     build_model
# )
sys.path.pop(0)


TRAIN_DATA_PERCENT_FOR_EVAL = 0.1
EVAL_ON_TRAIN_LOGS_NAME = (
    "random ({})-fraction"
    " of train data with frozen weights").format(
        TRAIN_DATA_PERCENT_FOR_EVAL
    )
WATERBIRDS_KEY = "waterbirds"
CAMELYON_17 = "camelyon17"
WILDS_DATASETS = (WATERBIRDS_KEY, CAMELYON_17)
UNLABELED_DATASET_KEY = "unlabeled_dataset"
TRAIN_SPLIT = "train"


def get_dataloaders(experiment_config, logger=None):

    data_config = experiment_config["data"]
    params_config = experiment_config["params"]
    dataset_config = data_config["dataset"]
    unlabeled_dataset_config = data_config.get(UNLABELED_DATASET_KEY)
    cache_path = experiment_config["cache_path"]
    num_readers = data_config["num_data_readers"]

    main_dataset_type = dataset_config["type"]

    eval_only_dataset_types = dataset_config.get("eval_only_types", [])
    additional_train_types = dataset_config.get("additional_train_types", [])

    list_of_train_loaders = []

    testloaders = {}
    trainloader = None

    for cur_dataset_type in (
        [main_dataset_type] + additional_train_types + eval_only_dataset_types
    ):

        cur_trainloader, cur_testloaders = get_dataloaders_for_type(
            cur_dataset_type,
            dataset_config,
            params_config,
            num_readers,
            cache_path,
            logger,
            eval_only=(cur_dataset_type in eval_only_dataset_types),
            train_only=(cur_dataset_type in additional_train_types)
        )

        if cur_trainloader is not None:
            list_of_train_loaders.append(cur_trainloader)

        if cur_testloaders is not None:
            testloaders |= cur_testloaders

    if len(list_of_train_loaders) == 0:
        trainloader = None
    elif len(list_of_train_loaders) == 1:
        trainloader = list_of_train_loaders[0]
    else:
        trainloader = chain_dataloaders(
            list_of_train_loaders,
            random_order=True
        )

    # add train subset into test dataloaders
    if trainloader is not None:

        eval_batch_size = trainloader.batch_size

        if len(testloaders) > 0:
            testloader = next(iter(testloaders.values()))
            if testloader is not None:
                eval_batch_size = testloader.batch_size

        if dataset_config.get("use_train_for_eval", True):
            testloaders[EVAL_ON_TRAIN_LOGS_NAME] \
                = randomly_subsampled_dataloader(
                    trainloader,
                    TRAIN_DATA_PERCENT_FOR_EVAL,
                    batch_size=eval_batch_size
                )

    assert trainloader or testloaders, \
        "Both trainloader and testloaders are None"

    if unlabeled_dataset_config is None:
        unlabeled_loaders = None
    else:
        unlabeled_dataset_type = get_with_assert(
            unlabeled_dataset_config,
            "type"
        )
        unlabeled_dataset_split = get_with_assert(
            unlabeled_dataset_config,
            "split"
        )

        split_is_train = (unlabeled_dataset_split == TRAIN_SPLIT)

        unlabeled_loaders = {}
        unlabeled_trainloader, unlabeled_testloaders = get_dataloaders_for_type(
            unlabeled_dataset_type,
            unlabeled_dataset_config,
            params_config,
            num_readers,
            cache_path,
            logger,
            eval_only=(not split_is_train),
            train_only=split_is_train,
            unlabeled=True
        )
        if unlabeled_dataset_split == TRAIN_SPLIT:
            unlabeled_loaders[unlabeled_dataset_split] = unlabeled_trainloader
        else:
            unlabeled_loaders[unlabeled_dataset_split] = get_with_assert(
                unlabeled_testloaders,
                unlabeled_dataset_split
            )

    return trainloader, testloaders, unlabeled_loaders


def get_dataloaders_for_type(
    dataset_type,
    dataset_config,
    params_config,
    num_readers,
    cache_path,
    logger,
    eval_only,
    train_only,
    unlabeled=False
):

    def assert_config(specific_dataset_config, dataset_type):
        assert isinstance(specific_dataset_config, dict)
        if dataset_type not in WILDS_DATASETS:
            assert len(specific_dataset_config), \
                f"Empty config for {dataset_type}"

    assert not (eval_only and train_only), \
        "eval_only and train_only cannot be both True"

    specific_dataset_config = dataset_config.get(dataset_type, {})
    assert_config(specific_dataset_config, dataset_type)

    sampler_config = dataset_config.get("sampler")

    if dataset_type != "features_labeller":
        assert sampler_config is None, \
            "Sampler is only supported for features labeller"

    transforms = make_transforms(specific_dataset_config.get("transforms"))
    eval_transforms = make_transforms(
        specific_dataset_config.get("eval_transforms")
    )

    if NAME_NUMBER_SEPARATOR in dataset_type:
        dataset_type, _ = parse_name_and_number(
            dataset_type,
            separator=NAME_NUMBER_SEPARATOR
        )

    eval_batch_size = train_batch_size = 0

    if unlabeled:
        assert params_config["to_train"], \
            "Unlabeled data should be used with \"to_train\" == True"
        batch_size = params_config["train"]["batch_size"]

        # don't create dataloader if it is not in active split
        if not train_only:
            eval_batch_size = batch_size
        if not eval_only:
            train_batch_size = batch_size
    else:

        if params_config["to_eval"] and not train_only:
            eval_batch_size = params_config["eval"]["batch_size"]

        if params_config["to_train"] and not eval_only:
            train_batch_size = params_config["train"]["batch_size"]

    # if dataset_type == "features_labeller":
    #     patch_features_labeller_config(specific_dataset_config)
    #     features_labeller = make_or_load_from_cache(
    #         "features_labeller_with_{}".format(
    #             specific_dataset_config["base_data"]["type"]
    #         ),
    #         specific_dataset_config,
    #         make_features_labeller,
    #         load_features_labeller,
    #         cache_path=cache_path,
    #         forward_cache_path=True,
    #         unique_hash=get_hash(
    #             {
    #                 key: value
    #                     for key, value
    #                     in specific_dataset_config.items()
    #                     if key not in ["off_diag_percent", "single_label"]
    #             }
    #         ),
    #         logger=logger
    #     )
    #     trainloaders, testloaders = features_labeller.get_dataloaders(
    #         train_batch_size,
    #         eval_batch_size,
    #         num_readers,
    #         off_diag_percent=specific_dataset_config.get("off_diag_percent", 0),
    #         single_label=specific_dataset_config.get("single_label", False),
    #         sampler_config=sampler_config
    #     )
    #     trainloader = trainloaders[features_labeller.diag_name]

    # elif dataset_type == TINY_IMAGENET:
    #     trainloader, testloaders \
    #         = get_tiny_imagenet_dataloaders(
    #             tiny_imagenet_config=specific_dataset_config,
    #             params_config=params_config,
    #             logger=logger
    #         )

    # elif dataset_type == "synthetic_pos":

    #     trainloader, testloaders \
    #         = get_synthetic_pos_dataloaders(
    #             dim_for_label=specific_dataset_config["dim_for_label"],
    #             num_train_images=specific_dataset_config["num_train_images"],
    #             num_test_images=specific_dataset_config["num_test_images"],
    #             train_batch_size=train_batch_size,
    #             test_batch_size=eval_batch_size,
    #             img_size=specific_dataset_config["img_size"],
    #             fill_object_with_ones\
    #                 =specific_dataset_config["fill_object_with_ones"]
    #         )

    # elif dataset_type in WILDS_DATASETS:
    if dataset_type in WILDS_DATASETS:
        trainloader, testloaders = get_wilds_dataloaders(
            dataset_type,
            train_batch_size=train_batch_size,
            test_batch_size=eval_batch_size,
            train_transform=transforms,
            test_transform=eval_transforms,
            fraction=specific_dataset_config.get("fraction", 1.0),
            root_dir=specific_dataset_config.get("root_dir"),
            unlabeled=specific_dataset_config.get("unlabeled", False),
            grouping_kwargs=specific_dataset_config.get("grouping_kwargs", {}),
            keep_metadata=specific_dataset_config.get("keep_metadata", True),
            use_waterbirds_divdis=specific_dataset_config.get(
                "use_waterbirds_divdis",
                False
            ),
            logger=logger
        )

    elif dataset_type == "imagenet1k":
    # if dataset_type == "imagenet1k":

        trainloader, testloaders = get_imagenet_dataloaders(
            train_batch_size,
            eval_batch_size,
            specific_dataset_config,
            train_transform=transforms,
            eval_transform=eval_transforms,
            return_index=specific_dataset_config.get("return_index", False),
            num_workers=num_readers
        )

    elif dataset_type == "model_vs_human":
        trainloader, testloaders = get_modelvshuman_dataloaders(
            specific_dataset_config,
            train_batch_size=train_batch_size,
            test_batch_size=eval_batch_size
        )

    # elif dataset_type == "from_folder":
    #     trainloaders, testloaders = get_dataloaders_from_folder(
    #         specific_dataset_config,
    #         train_batch_size,
    #         eval_batch_size,
    #         shuffle_train=True,
    #         shuffle_eval=False
    #     )
    #     if trainloaders is None:
    #         trainloader = None
    #     else:
    #         assert TRAIN_KEY in trainloaders
    #         trainloader = trainloaders[TRAIN_KEY]

    # elif dataset_type == "mnist_cifar":
    #     trainloader, testloaders = get_mnist_cifar_dataloaders(
    #         train_batch_size,
    #         eval_batch_size,
    #         mnist_cifar_config=specific_dataset_config,
    #         logger=logger
    #     )

    elif dataset_type == FROM_H5:
        trainloader, testloaders = get_h5_dataloaders(
            train_batch_size,
            eval_batch_size,
            from_h5_config=specific_dataset_config,
            num_workers=num_readers,
            logger=logger
        )

    # elif dataset_type == "easy_robust":
    elif dataset_type == "validation_only":
        assert train_batch_size == 0, \
            "train_batch_size should be 0 for easyrobust"
        # trainloader, testloaders = get_easy_robust_dataloaders(
        trainloader, testloaders = get_validation_only_dataloaders(
            train_batch_size,
            eval_batch_size,
            easy_robust_config=specific_dataset_config,
            num_workers=num_readers,
            eval_transform=eval_transforms,
            logger=logger
        )

    else:
        raise_unknown(
            "dataset type",
            dataset_type,
            "data config"
        )

    return trainloader, testloaders


# def patch_features_labeller_config(features_labeller_config):

#     base_data_config = get_with_assert(features_labeller_config, "base_data")
#     base_data_type = get_with_assert(base_data_config, "type")

#     if base_data_type == "dsprites":
#         pass
#     elif base_data_type == "utk_face":
#         features_labeller_config["make_base_data"] = make_utk_face_base_data
#     else:
#         raise_unknown(
#             "base data type",
#             base_data_type,
#             "patch_features_labeller_config"
#         )


# def get_manifold_projector_dataset(
#     projector_data_config,
#     cache_path,
#     logger=make_logger()
# ):
#     manifold_projector_dataset_type = projector_data_config["type"]

#     n_images = projector_data_config["total_number_of_images"]
#     assert n_images >= 0

#     projector_dataset_config \
#         = projector_data_config[manifold_projector_dataset_type]

#     if manifold_projector_dataset_type == TINY_IMAGENET_TEST:

#         return get_tiny_imagenet_test_projector_dataset(
#             manifold_projector_dataset_type,
#             projector_dataset_config,
#             n_images,
#             logger
#         )

#     elif manifold_projector_dataset_type == "dsprites":

#         return get_dsprites_unlabeled_data(
#             projector_dataset_config,
#             "train",
#             n_images,
#             cache_path,
#             logger
#         )

#     else:
#         raise_unknown(
#             "data for manifold projector",
#             manifold_projector_dataset_type,
#             "manifold projector data config"
#         )


# class FeatureExtractorTransform(torch.nn.Module):

#     def __init__(self, model_config, device=torch.device("cuda")):
#         super().__init__()
#         model = build_model(model_config)
#         model.eval()
#         self.feature_extractor, _ = separate_classifier_and_featurizer(model)
#         self.device = device
#         self.feature_extractor.to(self.device)

#     def forward(self, x):
#         with torch.no_grad():
#             return self.feature_extractor(
#                 x.to(self.device).unsqueeze(0)
#             ).squeeze(0).cpu()


def make_dataloaders(
    dataset_config,
    train_batch_size,
    eval_batch_size,
    num_workers,
    to_train=False,
    use_train_for_eval=False,
    logger=None
):

    assert "use_train_for_eval" not in dataset_config
    dataset_config["use_train_for_eval"] = use_train_for_eval
    exp_config = {
        "data": {
            "num_data_readers": num_workers,
            "dataset": dataset_config
        },
        "params": {
            "to_eval": True,
            "to_train": to_train,
            "eval": {
                "batch_size": eval_batch_size
            },
            "train": {
                "batch_size": train_batch_size
            },
        },
        "cache_path": make_default_cache_path()
    }
    return get_dataloaders(exp_config, logger=logger)


def make_hdf5_dataloader_from_path(
    cached_path,
    eval_batch_size,
    num_workers,
    max_chunk_size=MAX_CHUNK_SIZE,
    return_index=False,
    indices_to_keep_path=None,
    reverse_indices=False,
    total_samples=None
):
    split_name = "val"
    hdf5_config = {
        "type": "from_h5",
        "from_h5": {
            "return_index": return_index,
            "max_chunk": max_chunk_size,
            "keys_order": ["embed", "label"],
            "eval_splits": [split_name],
            "path": {
                split_name: cached_path
            },
            "indices_to_keep": {split_name: indices_to_keep_path}
        }
    }
    if total_samples is not None:
        hdf5_config["from_h5"]["total_samples"] = {
            split_name: total_samples
        }
    if indices_to_keep_path is not None:
        hdf5_config["from_h5"]["reverse_indices"] = {
            split_name: reverse_indices
        }

    _, val_dataloaders, _ = make_dataloaders(
        hdf5_config,
        0,
        eval_batch_size,
        num_workers
    )
    return val_dataloaders[split_name]


# def make_cached_dataloaders(dataloaders_dict, batch_size=32):
#     dataloaders = {}
#     for name, path in dataloaders_dict.items():
#         dataloaders[name] = make_hdf5_dataloader_from_path(
#             path,
#             eval_batch_size=batch_size,
#             num_workers=0  # should be 0 for hdf5
#         )
#     return dataloaders


def make_cached_dataloaders(
    dataloaders_dict,
    batch_size=32,
    max_chunk_size=MAX_CHUNK_SIZE
):
    dataloaders = {}
    for name, path in dataloaders_dict.items():
        dataloaders[name] = make_hdf5_dataloader_from_path(
            path,
            eval_batch_size=batch_size,
            max_chunk_size=max_chunk_size,
            num_workers=0  # should be 0 for hdf5
        )
    return dataloaders


def get_validation_only_dataloaders(
    train_batch_size,
    eval_batch_size,
    easy_robust_config,
    num_workers,
    eval_transform,
    logger,
):
    dataset_types = get_with_assert(easy_robust_config, "dataset_types")

    val_dataloaders = {}

    if eval_transform is None:
        eval_transform = make_default_test_transforms_imagenet()

    for dataset_type in dataset_types:
        assert dataset_type not in val_dataloaders, "Duplicate dataset type"
        if dataset_type in ["imagenet_a", "imagenet_r", "imagenet_v2"]:
            val_dataloaders[dataset_type] = get_imagenet_arv2_dataloader(
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
        elif dataset_type == "imagenet_c":
            val_dataloaders |= get_imagenet_c_dataloaders(
                eval_batch_size,
                easy_robust_config,
                num_workers,
                eval_transform,
                logger
            )
        elif dataset_type == "openimages":
            val_dataloaders[dataset_type] = get_openimages_dataloader(
                eval_batch_size,
                easy_robust_config,
                num_workers,
                eval_transform,
                logger
            )
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
