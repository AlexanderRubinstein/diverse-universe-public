from torch.utils.data import DataLoader
import argparse
import pickle
from torchvision.datasets import ImageFolder
import os
import sys
# from stuned.utility.utils import (
#     get_project_root_path
# )
from stuned.utility.utils import (
    raise_unknown
    # parse_list_from_string
)


# local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# import densifier
# from densifier.datasets.tardataset import TarDataset
# from densifier.get_segments.run_cropformer import EntityNetV2
from diverse_universe.local_datasets.common import (
    make_dataloaders
)
sys.path.pop(0)


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


def prepare_dataloaders(name, path):
    if name in ["in_train", "in_val"]:
        return make_in_dataloaders(name, path)
    else:
        raise_unknown(name, "dataset name", "prepare_dataloaders")


def make_in_dataloaders(name, path):
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
        to_train=True
        # to_train=False
    )
    if name == "in_train":
        return im_train_dataloader
    else:
        assert name == "in_val"
        return im_val_dataloaders["val"]
    # im_val_dataloader = im_val_dataloaders["val"]
    # im_train_dataloader = im_val_dataloaders["train"]


def main():
    # Loaded Arguments
    args = get_parser().parse_args()

    print(args) # tmp

    dataloader = prepare_dataloaders(
        args.original_dataset_name,
        args.original_dataset_path
    )


if __name__ == "__main__":
    # # Loaded Arguments
    # args = get_parser().parse_args()

    # print(args) # tmp

    # prepare_dataloaders(args.original_dataset_name, args.original_dataset_path)
    main()
