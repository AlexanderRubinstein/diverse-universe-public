# from torch.utils.data import DataLoader
import argparse
# import pickle
# from torchvision.datasets import ImageFolder
import os
import sys
from tqdm import tqdm
# from stuned.utility.utils import (
#     get_project_root_path
# )
import h5py
import numpy as np
import torch
# from stuned.utility.utils import (
#     raise_unknown,
#     get_hash,
#     get_with_assert,
#     remove_file_or_folder,
#     apply_random_seed,
#     # runcmd,
#     log_or_print
#     # parse_list_from_string
# )
# from stuned.utility.logger import make_logger


# local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# import densifier
# from densifier.datasets.tardataset import TarDataset
# from densifier.get_segments.run_cropformer import EntityNetV2
from diverse_universe.local_datasets.common import (
    make_dataloaders
)
from diverse_universe.local_models.common import (
    build_model,
    separate_classifier_and_featurizer
)
from diverse_universe.local_datasets.from_h5 import (
    make_hdf5_name,
    extract_hdf5_hash
)
from diverse_universe.local_datasets.easy_robust import (
    download_in_a,
    download_in_r,
    # download_oi,
    # get_imagenet_arv2_dataloader
    get_easy_robust_dataloaders
)
from diverse_universe.local_datasets.openimages import (
    get_openimages_dataloader,
    download_oi
)
# from diverse_universe.local_datasets.from_folder import (
#     get_from_folder_dataloader
# )
from diverse_universe.local_datasets.inaturalist import (
    download_inat,
    get_inat_dataloader
)
from diverse_universe.local_datasets.imagenet_c import (
    IN_C_DATALOADERS_NAMES,
    IN_C_URL,
    get_imagenet_c_dataloaders
)
# from diverse_universe.local_datasets.utils import (
#     download_tar_from_gdrive
# )
sys.path.pop(0)


from stuned.utility.utils import (
    raise_unknown,
    get_hash,
    get_with_assert,
    remove_file_or_folder,
    apply_random_seed,
    # runcmd,
    log_or_print
    # parse_list_from_string
)
from stuned.utility.logger import make_logger
from stuned.local_datasets.transforms import (
    make_default_test_transforms_imagenet
)


# def pop_arg_from_opts(args, arg_name):
#     cutoff_i = None
#     output = None
#     for i in range(len(args.opts)):
#         if args.opts[i] == arg_name:
#             output = args.opts[i+1]
#             cutoff_i = i
#             break
#     assert cutoff_i is not None
#     args.opts = args.opts[:cutoff_i] + args.opts[cutoff_i+2:]
#     return output


def get_parser():
    parser = argparse.ArgumentParser(description="Cache datasets")
    parser.add_argument(
        "--model",
        help="Model to use as feature extractor",
        choices=["deit3b"]
    )
    parser.add_argument(
        "--model_path",
        help="Path to the feature extractor"
    )
    parser.add_argument(
        "--layer_cutoff",
        help="Till which layer to extract features"
    )
    parser.add_argument(
        "--original_dataset_name",
        help="Name of dataset to cache"
    )
    parser.add_argument(
        "--original_dataset_path",
        help="Path to the dataset to cache"
    )
    parser.add_argument(
        "--cache_save_path",
        help="Where to save cached features"
    )
    parser.add_argument(
        "--n_epochs",
        help="How many epochs to cache (makes sense for random augmentations)"
    )
    # parser.add_argument("--input", help="Path to a tar file")
    # parser.add_argument("--range", help="Range of images to process", default=None)
    # parser.add_argument("--output", help="A directory to save output predictions dump.")
    # parser.add_argument("--confidence-threshold",type=float,default=0.5,help="Minimum score for instance predictions to be shown")
    # parser.add_argument(
    #     "--opts",
    #     help="Modify config options using the command-line 'KEY VALUE' pairs",
    #     default=[],
    #     nargs=argparse.REMAINDER)
    return parser

    # ??

    # if args.output is None:
    #     args.output = pop_arg_from_opts(args, "--output")

    # if args.range is None:
    #     args.range = pop_arg_from_opts(args, "--range")

    # print("Arguments: " + str(args))

    # # Load Dataset
    # input_path = args.input
    # if input_path.endswith('.tar') or input_path.endswith('.tar.gz'):
    #     dataset = TarDataset(input_path, transform=None)
    # else:
    #     dataset = ImageFolder(input_path, transform=None)
    # if args.range is None:
    #     images_range = None
    # else:
    #     # assert ":" == args.range[0]
    #     # assert ":" == args.range[1]
    #     # args.range = args.range[1:-1]
    #     # images_range = parse_list_from_string(args.range, list_separators=":")
    #     start, end = args.range.split(':')
    #     images_range = [int(start), int(end)]

    # # Extract Segments
    # net = EntityNetV2(args)
    # output = net.run(range=images_range)

    # # Save Segments
    # os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # pickle.dump(output, open(args.output, 'wb'))


def empty_path(path):
    return not os.path.exists(path) or len(os.listdir(path)) == 0


def make_in_c_dataloaders(
    path,
    corruption_types=list(IN_C_DATALOADERS_NAMES.keys()),
    severity_levels=[1, 5],
    batch_size=128,
    num_workers=4,
    logger=None
):
    # imagenet_c_config = {
    #     "type": "easy_robust",
    #     'easy_robust': {
    #         "dataset_types": ["imagenet_c"],
    #         # "data_dir": "./tmp/imagenet-a.tar.gz"
    #         "data_dir": "/mnt/qb/work/oh/arubinstein17/cache/ImageNet-C/imagenet-c",
    #         # "inc_types": ["Fog"]
    #         # "inc_types": None
    #         "inc_types": [
    #             'Brightness',
    #             'Contrast',
    #             'Defocus Blur',
    #             'Elastic Transform',
    #             'Fog',
    #             'Frost',
    #             'Gaussian Noise',
    #             'Glass Blur',
    #             'Impulse Noise',
    #             'JPEG Compression',
    #             'Motion Blur',
    #             'Pixelate',
    #             'Shot Noise',
    #             'Snow',
    #             'Zoom Blur',
    #         ]
    #     }
    # }
    if empty_path(path):
        raise FileNotFoundError(
            f"ImageNet-C path {path} does not exist or is empty. "
            f"Please download ImageNet-C manually, "
            f"e.g. from here (48GB): {IN_C_URL}"
        )

    in_c_config = {
            # "data_dir": "/mnt/qb/work/oh/arubinstein17/cache/ImageNet-C/imagenet-c",
            "data_dir": path,
            # "inc_types": ["Fog"]
            # "inc_types": None
            "inc_types": corruption_types,
            # [
            #     'Brightness',
            #     'Contrast',
            #     'Defocus Blur',
            #     'Elastic Transform',
            #     'Fog',
            #     'Frost',
            #     'Gaussian Noise',
            #     'Glass Blur',
            #     'Impulse Noise',
            #     'JPEG Compression',
            #     'Motion Blur',
            #     'Pixelate',
            #     'Shot Noise',
            #     'Snow',
            #     'Zoom Blur',
            # ]
    }

# _, imagenet_c_dataloaders, _ = make_dataloaders(
#     imagenet_c_config,
#     train_batch_size=0,
#     eval_batch_size=128,
#     num_workers=4
# )

# imagenet_c_dataloaders = {key: value for key, value in imagenet_c_dataloaders.items() if ("1" in key or "5" in key)}
    imagenet_c_dataloaders = get_imagenet_c_dataloaders(
        eval_batch_size=batch_size,
        in_c_config=in_c_config,
        num_workers=num_workers,
        eval_transform=None,
        logger=logger,
        severity_levels=severity_levels
    )
    return imagenet_c_dataloaders


# ?? don't use as is for IN-C

def make_inat_dataloader(path, batch_size=128, num_workers=4, logger=None):
#                             inaturalist_config = {
#     "type": "easy_robust",
#     'easy_robust': {
#         "dataset_types": ["from_folder"],
#         "data_dir": "/mnt/qb/work/oh/arubinstein17/cache/iNaturalist/iNaturalist/",
#         "dataset_name": "iNaturalist"
#     }
# }

# _, easy_robust_dataloaders, _ = make_dataloaders(
#     inaturalist_config,
#     train_batch_size=0,
#     eval_batch_size=128,
#     num_workers=4
# )
# inat_dataloader = easy_robust_dataloaders["iNaturalist"]
# if eval_transform is None:
#     eval_transform = make_default_test_transforms_imagenet()d
    data_dir = os.path.join(path, "iNaturalist")
    download_if_not_exists(data_dir, download_inat)
    inat_config = {
        # "dataset_types": ["from_folder"],
        # "data_dir": "/mnt/qb/work/oh/arubinstein17/cache/iNaturalist/iNaturalist/",
        # "data_dir": os.path.join(data_dir, "images"),
        "data_dir": data_dir,
        "dataset_name": "iNaturalist"
    }
    eval_transform = make_default_test_transforms_imagenet()

    inat_dataloader = get_inat_dataloader(
        # train_batch_size,
        eval_batch_size=batch_size,
        inat_config=inat_config,
        num_workers=num_workers,
        eval_transform=eval_transform,
        logger=logger
    )
    return inat_dataloader


def make_oi_dataloader(path, batch_size=128, num_workers=4, logger=None):

    data_dir = os.path.join(path, "openimages")
    # img_list = os.path.join(path, "openimage_o.txt")
    # download_if_not_exists(data_dir, download_oi)
    download_if_not_exists(data_dir, download_oi)
    # assert os.path.exists(img_list)
    # list_path = os.path.join(path, "openimage_o.txt")

    openimages_config = {
        # "type": "easy_robust",
        # 'easy_robust': {
        #     "dataset_types": ["openimages"],
        #     "data_dir": data_dir,
        #     "img_list": None
        # }
        "data_dir": data_dir,
        "img_list": None
    }
    # ??
    oi_dataloader = get_openimages_dataloader(
        eval_batch_size=batch_size,
        oi_config=openimages_config,
        num_workers=num_workers,
        eval_transform=None,
        logger=logger
    )
    # _, easy_robust_dataloaders, _ = make_dataloaders(
    #     openimages_config,
    #     train_batch_size=0,
    #     eval_batch_size=128,
    #     num_workers=4
    # )
    # openimages_dataloader = easy_robust_dataloaders["openimages"]
    return oi_dataloader


def download_if_not_exists(data_dir, download_func):
    if empty_path(data_dir):
        # os.makedirs(path, exist_ok=True)
        download_func(data_dir)


def make_in_ar_dataloader(
    name,
    path,
    batch_size=128,
    num_workers=4,
    logger=None
):

    if name == "in_a":
        dataset_type = "imagenet_a"
        download_func = download_in_a
    # elif name == "openimages":
    #     dataset_type = name
    #     download_func = download_oi
    else:
        assert name == "in_r"
        dataset_type = "imagenet_r"
        download_func = download_in_r

    data_dir = os.path.join(path, dataset_type.replace("_", "-"))
    # if not os.path.exists(expected_path) or len(os.listdir(expected_path)) == 0:
    #     os.makedirs(path, exist_ok=True)
    #     download_func(path)
    download_if_not_exists(data_dir, download_func)

#     openimages_config = {
#     "type": "easy_robust",
#     'easy_robust': {
#         "dataset_types": ["openimages"],
#         "data_dir": "/mnt/qb/work/oh/arubinstein17/cache/OpenImage/test",
#         "img_list": "/mnt/qb/work/oh/arubinstein17/cache/OpenImage/openimage_o.txt"
#     }
# }

# _, easy_robust_dataloaders, _ = make_dataloaders(
#     openimages_config,
#     train_batch_size=0,
#     eval_batch_size=128,
#     num_workers=4
# )
# openimages_dataloader = easy_robust_dataloaders["openimages"]

    imagenet_a_config = {
        # "type": "easy_robust",
        # 'easy_robust': {
        #     "dataset_types": [dataset_type],
        #     # "data_dir": "./tmp/imagenet-a.tar.gz"
        #     # "data_dir": "/mnt/lustre/work/oh/arubinstein17/cache/imagenet-a"
        #     "data_dir": path
        # }
        "dataset_types": [dataset_type],
        "data_dir": data_dir
    }

    # _, easy_robust_dataloaders, _ = make_dataloaders( ??
    _, easy_robust_dataloaders = get_easy_robust_dataloaders(
        train_batch_size=0,
        eval_batch_size=batch_size,
        easy_robust_config=imagenet_a_config,
        num_workers=num_workers,
        eval_transform=None,
        logger=logger
    )
    imagenet_a_dataloader = easy_robust_dataloaders[dataset_type]
    return imagenet_a_dataloader


def prepare_dataloaders(name, path, logger):

    if name in ["in_train", "in_val"]:
        return make_in_dataloaders(name, path, logger=logger)
    elif name in ["in_a", "in_r"]:
        return make_in_ar_dataloader(name, path, logger=logger)
    elif name == "inaturalist":
        return make_inat_dataloader(path, logger=logger)
    elif name == "openimages":
        return make_oi_dataloader(path, logger=logger)
    elif name == "in_c":
        return make_in_c_dataloaders(path, logger=logger)
    else:
        raise_unknown(name, "dataset name", "prepare_dataloaders")


def make_in_dataloaders(name, path, logger):
    imagenet_config = {
        "type": "imagenet1k",
        "imagenet1k": {
            "only_val": False,
            # "hf_token_path": DEFAULT_HUGGING_FACE_ACCESS_TOKEN_PATH,
            # "cache_dir": "./tmp/",
            # "data_dir": None,
            # "test_splits": ["val", "train"],
            "test_splits": ["val"],
            "train_split": "train",
            # "train_split": "train",
            # "path": "/mnt/beegfs/oh/datasets/ImageNet/ILSVRC/Data/CLS-LOC",
            # "path": "/scratch_local/arubinstein17-51549/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC",
            # "path": "/mnt/lustre/datasets/ImageNet2012",
            "path": path,
            "use_train_for_eval": False,
            # "num_test_samples": 1000, # debug
            # "num_train_samples": 1000, # debug
            "transforms": {
                "transforms_list": [
                    "from_class-RRC",
                    "from_class-RandomHorFlip",
                    "from_class-ToTensor",
                    "from_class-Normalize",
                ],
                "from_class-Normalize": {
                    "class": "torchvision.transforms.Normalize",
                    "kwargs": {
                        "std": [
                            0.229,
                            0.224,
                            0.225
                        ],
                        "mean": [
                            0.485,
                            0.456,
                            0.406
                        ]
                    }
                },
                "from_class-ToTensor": {
                    "class": "torchvision.transforms.ToTensor"
                },
                "from_class-RRC":  {
                    "class": "torchvision.transforms.RandomResizedCrop",
                    "kwargs": {
                        "scale": (0.08, 1.0),
                        "ratio": (0.75, 1.3333333333333333),
                        "size": 224
                    }
                },
                "from_class-RandomHorFlip": {
                "class": "torchvision.transforms.RandomHorizontalFlip",
                "kwargs": {}
                }
            },
            "eval_transforms": {
                "transforms_list": [
                    "from_class-Resize",
                    "from_class-CenterCrop",
                    "from_class-ToTensor",
                    "from_class-Normalize"
                ],
                "from_class-Normalize": {
                    "class": "torchvision.transforms.Normalize",
                    "kwargs": {
                        "std": [
                        0.229,
                        0.224,
                        0.225
                        ],
                        "mean": [
                        0.485,
                        0.456,
                        0.406
                        ]
                    }
                },
                "from_class-Resize": {
                    "class": "torchvision.transforms.Resize",
                    "kwargs": {
                        "size": 256
                    }
                },
                "from_class-ToTensor": {
                    "class": "torchvision.transforms.ToTensor"
                },
                "from_class-CenterCrop": {
                    "class": "torchvision.transforms.CenterCrop",
                    "kwargs": {
                        "size": 224
                    }
                },
            }
        }
    }

    batch_size = 128
    num_workers = 4

    im_train_dataloader, im_val_dataloaders, _ = make_dataloaders(
        imagenet_config,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        to_train=True,
        logger=logger
        # to_train=False
    )
    # im_train_dataloader, im_val_dataloaders = get_imagenet_dataloaders(
    #     train_batch_size,
    #     eval_batch_size,
    #     specific_dataset_config,
    #     train_transform=transforms,
    #     eval_transform=eval_transforms,
    #     return_index=specific_dataset_config.get("return_index", False),
    #     num_workers=num_readers
    # )
    if name == "in_train":
        return im_train_dataloader
    else:
        assert name == "in_val"
        return im_val_dataloaders["val"]
    # im_val_dataloader = im_val_dataloaders["val"]
    # im_train_dataloader = im_val_dataloaders["train"]


def prepare_model_config(name, path):
    if name == "deit3b":
        config = {
            "type": "torch_load",
            "torch_load": {
                "path": path,
                "model_name": "deit3_21k"
            }
        }
    return config


def save_activations(
    dataset, # can be dataloader or dataset
    cache_file_path,
    n_epochs,
    total_samples,
    dataset_config, # can be dict or str if name
    model_config,
    batch_size=512,
    num_workers=4,
    random_seed=42,
    collate_fn=None,
    wrap=None,
    block_id=None,
    custom_prefix=None,
    use_path_as_is=False,
    logger=None
):

    if isinstance(dataset_config, dict):
        unique_hash = get_hash(dataset_config | model_config)
        dataset_type = get_with_assert(dataset_config, "type")
    else:
        dataset_type = dataset_config
        unique_hash = get_hash(model_config | {"dataset_type": dataset_type})

    # in case of SingleDataloaderWrapper
    if hasattr(dataset, "dataloader"):
        dataset = dataset.dataloader

    if isinstance(dataset, torch.utils.data.DataLoader):
        dataset = dataset.dataset
        # assert batch_size == dataloader.batch_size
    # else:
    device = torch.device("cuda:0")

    model = build_model(model_config)

    model_type = get_with_assert(model_config, "type")

    # dataset_type = get_with_assert(dataset_config, "type")

    model.eval()

    model, classifier = separate_classifier_and_featurizer(
        model,
        block_id=block_id
    )

    model.to(device)

    # input_0, _ = dataset[0]
    dataset_item = dataset[0]
    # print(dataset_item)  # debug
    if collate_fn is None:
        input_0 = dataset_item[0].unsqueeze(0)
    else:
        input_0 = collate_fn([dataset_item])
        # print(input_0)  # debug
        input_0 = input_0[0]
        # print(input_0.shape)  # debug
    embed_shape = model(input_0.to(device)).shape[-1]

    if total_samples == 0:
        total_samples = len(dataset)

    total_batches = total_samples // batch_size
    last_batch_size = total_samples % batch_size
    if last_batch_size != 0:
        total_batches += 1
    else:
        last_batch_size = batch_size

    if block_id is not None:
        model_type_name = model_type + f"_block_{block_id}"
    else:
        model_type_name = model_type

    if custom_prefix is not None:
        model_type_name = f"{custom_prefix}_{model_type_name}"

    if not use_path_as_is:
        cache_file_path = os.path.join(
            cache_file_path,
            make_hdf5_name(
                unique_hash,
                f"{model_type_name}_model_{dataset_type}_dataset_{n_epochs}_epochs_{total_samples}_samples"
            )
        )

    log_or_print(f"cache_file_path: {cache_file_path}", logger)

    if os.path.exists(cache_file_path):
        overwrite = False
        try:
            with h5py.File(cache_file_path, 'r') as hdf5_file:
                # print(hdf5_file.keys())
                # print(hdf5_file)
                # assert False
                if "embed" not in hdf5_file or np.array(hdf5_file["embed"])[-1].sum() == 0:
                    overwrite = True
        except (BlockingIOError, OSError):
            overwrite = True

        if overwrite:
            log_or_print(
                f"cache path {cache_file_path} already exists \n"
                f"But it is empty, so it will be overwritten",
                logger
            )
            remove_file_or_folder(cache_file_path)
        else:


            log_or_print(
                f"cache path {cache_file_path} already exists \n"
                f"Skipping {dataset_type}",
                logger
            )
            return cache_file_path
            # os.remove(cache_file_path)
            # print(f"File {cache_file_path} already exists")

    samples_after_repetition = total_samples * n_epochs

    os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)

    # assert not hasattr(dataset, "dataset"), "dataset should not be a Subset"
    if hasattr(dataset, "dataset"):
        log_or_print("Dataset is a Subset", logger)

    start_index = 0

    apply_random_seed(random_seed)
    # if isinstance(dataset, torch.utils.data.DataLoader):
    #     dataloader = dataset
    #     assert batch_size == dataloader.batch_size
    # else:
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # if wrap is not None:
    # if wrap == "mvh":
    #     dataloader = wrap_mvh_dataloader(dataloader)
    # else:
    #     assert wrap is None, "wrap should be None, 'mvh' or 'none'"

    with h5py.File(cache_file_path, 'w') as hdf5_file:
        hdf5_file.create_dataset(
            "embed",
            shape=(samples_after_repetition, embed_shape),
            maxshape=(None, embed_shape)
        )
        hdf5_file.create_dataset(
            "label",
            shape=(samples_after_repetition, 1),
            maxshape=(None, 1)
        )
        hdf5_file.create_dataset(
            "index",
            shape=(samples_after_repetition, 1),
            maxshape=(None, 1)
        )

        for j in range(n_epochs):
            # apply_random_seed(j)
            # dataloader = torch.utils.data.DataLoader(
            #     dataset,
            #     batch_size=batch_size,
            #     shuffle=False,
            #     num_workers=num_workers
            # )

            for i, dataloader_item in tqdm(enumerate(dataloader)):
                input, target = dataloader_item[0], dataloader_item[1]

                with torch.no_grad():

                    features = model(input.to(device))

                seen_samples = j * total_samples
                if i + 1 == total_batches:
                    end_index = start_index + last_batch_size
                    features = features[:last_batch_size]
                    target = target[:last_batch_size]
                else:
                    end_index = start_index + input.shape[0]
                indices_range = list(
                    range(start_index - seen_samples, end_index - seen_samples)
                )

                hdf5_file["embed"][start_index:end_index, :] = features.cpu().numpy()
                hdf5_file["label"][start_index:end_index, 0] = target
                hdf5_file["index"][start_index:end_index, 0] = indices_range

                start_index = end_index
                if i + 1 == total_batches:
                    break

    return cache_file_path


def cache_dataloaders(
    dataloaders_dict,
    cache_path,
    model_config,
    batch_size=128,
    num_workers=4,
    block_id=None,
    collate_fn=None,
    custom_prefix=None,
    use_path_as_is=False,
    logger=None
):
    res = {}
    for dataloader_name, dataloader in dataloaders_dict.items():
        if use_path_as_is:
            cur_cache_path = cache_path
        else:
            cur_cache_path = os.path.join(cache_path, dataloader_name)

        collate_fn = collate_fn
        wrap = None
        if isinstance(dataloader, tuple):
            dataloader, wrap = dataloader
        print("Caching dataloader: ", dataloader_name)
        res[dataloader_name] = save_activations(
            dataloader,
            cur_cache_path,
            n_epochs=1,
            total_samples=0,
            dataset_config=dataloader_name,
            model_config=model_config,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
            wrap=wrap,
            block_id=block_id,
            custom_prefix=custom_prefix,
            use_path_as_is=use_path_as_is,
            logger=logger
        )
    return res


def main():
    # Loaded Arguments
    args = get_parser().parse_args()
    logger = make_logger()

    # print(args) # tmp

    dataset_name = args.original_dataset_name
    dataloader = prepare_dataloaders(
        dataset_name,
        args.original_dataset_path,
        logger
    )
    if isinstance(dataloader, dict):
        use_path_as_is = False
        dataloaders_dict = dataloader
    else:
        use_path_as_is = True
        dataloaders_dict = {dataset_name: dataloader}

    model_config = prepare_model_config(
        args.model,
        args.model_path
    )

    cache_dataloaders(
        # {
        #     "iNaturalist": inat_dataloader,
        #     "openimages": openimages_dataloader,
        #     # "im_train_dataloader": im_train_dataloader,
        #     # "im_val_dataloader": im_val_dataloader
        # }
        #     # | imagenet_c_dataloaders
        #     | all_dataloaders_dict,
        dataloaders_dict,
        # tmp_dict,
        # VAL_DATASETS_CACHE,
        # ?? cache path
        args.cache_save_path,
        model_config,
        block_id=int(args.layer_cutoff),
        use_path_as_is=use_path_as_is,
        logger=logger
        # custom_prefix="deit3_2layer_straight_order"
    )


if __name__ == "__main__":
    # # Loaded Arguments
    # args = get_parser().parse_args()

    # print(args) # tmp

    # prepare_dataloaders(args.original_dataset_name, args.original_dataset_path)
    main()
