import torch
import math
import matplotlib.pyplot as plt
from typing import Iterable
import sys
import os
import torch
from tqdm import tqdm
import itertools
import gdown
from torchvision import transforms
# import torch.nn.functional as F
import numpy as np
from stuned.utility.utils import (
    get_project_root_path,
    load_from_pickle,
    # aggregate_tensors_by_func,
    get_even_from_wrapped,
    error_or_print,
    run_cmd_through_popen,
    remove_file_or_folder,
    get_current_time
)


# # local modules
# sys.path.insert(
#     0,
#     # os.path.join(
#     #     os.path.dirname(os.path.abspath('')), "src"
#     # )
#     get_project_root_path
# )
# from utility.utils import (
#     # get_even_from_wrapped,
#     # load_from_pickle,
#     # aggregate_tensors_by_func,
#     # get_project_root_path
# )
# sys.path.pop(0)


FONT_SIZE = 48
JSON_PATH = os.path.join(
    get_project_root_path(),
    "diverse_universe",
    "json"
)
SHARING_SUFFIX = "/view?usp=sharing"
# ADJUSTED_GROUPWISE_KEY = "_adjusted"


class DatasetWrapperWithIndex(torch.utils.data.Dataset):

        def __init__(self, dataset):
            self.dataset = dataset

        def __getitem__(self, index):
            return *self.dataset[index], index

        def __len__(self):
            return len(self.dataset)


def make_dataset_wrapper_with_index(dataset):
    return DatasetWrapperWithIndex(dataset)


# TODO(Alex | 19.12.2023) refactor functions:
# show_image, show_images_container, show_images_from_dataloader
def show_image(
    image,
    label_dict,
    axis,
    unnormalizer=None,
    label_mapper=(lambda x: x),
    font_size=FONT_SIZE
):

    if unnormalizer is not None:
        image = unnormalizer(image.unsqueeze(0))[0]

    # Convert the image from a tensor to a numpy array and transpose it
    image = image.numpy().transpose((1, 2, 0))

    # Plot the image and its label
    axis.imshow(image)
    if label_dict is None:
        caption = "No label"
    else:
        caption = '\n'.join(
            [
                f"{label_name}: {label_mapper(label.item())}"
                    for label_name, label
                        in label_dict.items()
            ]
        )

    axis.set_title(
        caption,
        fontsize=font_size
    )
    axis.axis("off")


def show_images_container(images, labels_dict, **show_kwargs):
    n_images = len(images)
    n = int(math.sqrt(n_images))
    m = n_images // n + (n_images % n != 0)
    fig, axes = plt.subplots(nrows=m, ncols=n, figsize=(8 * m, 8 * n))

    flat = (n == 1 or m == 1)

    i = 0

    if not isinstance(axes, Iterable):
        axes = [axes]

    for axes_array in axes:

        if flat:
            axes_array = [axes_array]

        for axis in axes_array:
            if i == len(images):
                break
            if labels_dict is None:
                label_dict = None
            else:
                label_dict = {
                    label_name: label_batch[i]
                        for label_name, label_batch
                        in labels_dict.items()
                }

            show_image(
                images[i],
                label_dict,
                axis,
                **show_kwargs
            )
            i += 1

    fig.tight_layout()
    fig.show()
    return fig


def show_images_from_dataloader(
    dataloader,
    n,
    label_names=["label"],
    batch_id=0,
    in_case_of_multilable=None,
    model=None,
    **show_kwargs
):

    def show_batch(
        dataloader,
        dataloader_item,
        n,
        label_names,
        in_case_of_multilable,
        model=None,
        **show_kwargs
    ):
        if get_even_from_wrapped(dataloader, "dataloader", "has_metadata"):
            dataloader_item = dataloader_item[:-1]
        images = dataloader_item[0][:n]
        if label_names is None:
            labels_dict = None
        else:
            if in_case_of_multilable == "first":
                assert len(label_names) == 1
                assert len(dataloader_item) > 1
                dataloader_item = dataloader_item[:2]
            assert len(label_names) + 1 == len(dataloader_item)
            labels_dict = {}
            for labels, labels_name in zip(dataloader_item[1:], label_names):
                labels_dict[labels_name] = labels[:n]

        if model is not None:
            model.to(images.device)
            model.eval()
            with torch.no_grad():
                pred = model(images)
            labels_dict["pred"] = torch.argmax(pred, dim=-1)[:n]

        show_images_container(images, labels_dict, **show_kwargs)

    # show at most "n" images from "batch_id"-th batch
    dataloader_item = None
    total_batches = len(dataloader)
    if not isinstance(batch_id, list):
        batch_id = [batch_id]
    batch_id = set(batch_id)
    biggest_batch_id = max(batch_id)
    if biggest_batch_id >= total_batches:
        raise ValueError(
            f"Batch {biggest_batch_id} does not exist. "
            f"Only {total_batches} batches are available."
        )

    shown_counter = 0
    for i, batch in enumerate(dataloader):
        if i in batch_id:
            dataloader_item = batch
            show_batch(
                dataloader,
                dataloader_item,
                n,
                label_names,
                in_case_of_multilable,
                model=model,
                **show_kwargs
            )
            shown_counter += 1
        if shown_counter == len(batch_id):
            break


def extract_subset_indices(subset_indices, split):
    path_to_indices = subset_indices.get(split)
    if path_to_indices is None:
        return None
    else:
        if path_to_indices.split('.')[-1] == "pt":
            return torch.load(path_to_indices)
        return load_from_pickle(path_to_indices)


# PER_SAMPLE_METRIC_NAMES = [
#     "div_different_preds_per_sample",
#     "div_continous_unique_per_sample"
# ]


# def record_diversity(
#     res,
#     outputs,
#     stacked_outputs,
#     metrics_mappings,
#     labels=None,
#     name_prefix="",
#     detailed_results=None
# ):

#     # metrics_mappings is a tuple of tuples:
#     # ((name_1, func_1), ... (name_k, func_k))
#     for metric_tuple in metrics_mappings:

#         metric_name = metric_tuple[0]
#         metric_key = name_prefix + metric_name
#         compute_metric = metric_tuple[1]
#         if metric_name not in res:
#             res[metric_key] = 0
#         if metric_name in PER_SAMPLE_METRIC_NAMES:
#             value = compute_metric(stacked_outputs)
#         elif metric_name == "div_ortega":
#             assert labels is not None
#             value = compute_metric(stacked_outputs, labels).item()
#         elif metric_name in [
#             "var",
#             "std",
#             "dis",
#             "max_var",
#             "div_different_preds",
#             "div_mean_logits",
#             "div_max_logit",
#             "div_entropy",
#             "div_max_prob",
#             "div_mean_prob",
#             "div_different_preds_per_model",
#             "div_continous_unique"
#         ]:
#             value = compute_metric(stacked_outputs).item()
#         else:
#             value = aggregate_tensors_by_func(
#                 apply_pairwise(outputs, compute_metric)
#             ).item()

#         if not torch.is_tensor(value):
#             res[metric_key] += value

#         if detailed_results is not None:
#             if metric_key in PER_SAMPLE_METRIC_NAMES:
#                 if metric_key not in detailed_results:
#                     detailed_results[metric_key] = []

#                 if metric_key in detailed_results:
#                     for subvalue in value:
#                         detailed_results[metric_key].append(subvalue.item())


# def are_probs(logits):
#     if (
#             logits.min() >= 0
#         and
#             logits.max() <= 1
#         # don't check sums to one for the cases
#         # like IN_A where masking drops some probs

#         # and
#         #     abs(logits.sum(-1)[0][0] - 1) > EPS
#     ):
#         return True
#     return False


# def get_probs(logits):
#     if are_probs(logits):
#         probs = logits
#     else:
#         probs = F.softmax(logits, dim=-1)
#     return probs


# # TODO(Alex | 15.05.2024): Make it readable
# def evaluate_model(
#     model,
#     dataloader,
#     device=torch.device("cuda:0"),
#     select_output=None,
#     metrics_mappings=None,
#     feature_extractor=None,
#     return_detailed=False,
#     prune_metrics=["dis"],
#     metadata_to_group=None,
#     logger=None,
#     **evaluation_kwargs
# ):

#     def update_correct(
#         res,
#         key,
#         outputs,
#         labels,
#         detailed_results,
#         metadata,
#         per_group_totals
#     ):

#         def add_to_dict(dict, key, subkey, value):
#             if key not in dict:
#                 dict[key] = {}
#             if subkey not in dict[key]:
#                 dict[key][subkey] = []
#             dict[key][subkey].append(value)

#         def zero_key(d, k):
#             if k not in d:
#                 d[k] = 0

#         def pad(arr, bigger_arr, value=0):
#             while len(arr) < len(bigger_arr):
#                 if isinstance(arr, np.ndarray):
#                     arr = np.append(arr, [value])
#                 else:
#                     assert isinstance(arr, list)
#                     arr.append(value)
#             return arr

#         zero_key(res, key)

#         # masking out samples from ImageNet-A/R
#         # that did not have argmax within selected 200 classes
#         mask = (outputs.sum(-1) == 0)
#         predicted = torch.argmax(outputs, dim=-1)
#         predicted[mask] = -1

#         if detailed_results is not None:
#             probs = get_probs(outputs)
#             for i, pred in enumerate(predicted):
#                 pred = pred.item()
#                 add_to_dict(detailed_results, key, "pred", pred)
#                 add_to_dict(detailed_results, key, "conf", probs[i, pred].item())

#         if len(predicted.shape) == 1 and len(labels.shape) == 2:
#             # ImageNet-hard case, based on this: https://github.com/kirill-vish/Beyond-INet/blob/fd9b1b6c36ecf702fbcc355e037d8e9d307b0137/inference/robustness.py#L117C9-L117C71
#             assert predicted.shape[0] == labels.shape[0]
#             res[key] += (predicted[:, None] == labels).any(1).sum().item()
#         else:
#             assert predicted.shape == labels.shape
#             correct = (predicted == labels)

#             # compute stats for different groups
#             if metadata is not None:

#                 assert metadata_to_group is not None
#                 groups = metadata_to_group(metadata)

#                 max_group = groups.max().item()
#                 num_groups = max_group + 1

#                 group_acc = np.array([0.0] * num_groups)
#                 group_count = np.array([0.0] * num_groups)

#                 denom = []
#                 processed_data_counts = []

#                 for current_group in range(num_groups):
#                     mask = (groups == current_group).to(correct.device)

#                     group_key = key + f'_group_{current_group}'

#                     zero_key(res, group_key)
#                     zero_key(per_group_totals, group_key)

#                     current_group_count = mask.sum().item()
#                     group_count[current_group] \
#                         = current_group_count
#                     num_correct_for_group = (mask * correct).sum().item()
#                     group_acc[current_group] \
#                         = num_correct_for_group / (current_group_count + int(current_group_count == 0))

#                     processed_data_counts.append(per_group_totals[group_key])
#                     per_group_totals[group_key] += group_count[current_group]  # for unweighted group accuracy
#                     denom.append(per_group_totals[group_key])
#                     res[group_key] += num_correct_for_group  # for unweighted group accuracy

#                 group_wise_key = key + ADJUSTED_GROUPWISE_KEY
#                 if group_wise_key not in res:
#                     res[group_wise_key] = np.array([0] * num_groups)
#                 else:
#                     res_groupwise = res[group_wise_key]
#                     processed_data_counts = pad(
#                         processed_data_counts,
#                         res_groupwise,
#                         1
#                     )
#                     denom = pad(denom, res_groupwise)
#                     group_acc = pad(group_acc, res_groupwise)
#                     group_count = pad(group_count, res_groupwise)

#                 denom = np.array(denom)
#                 processed_data_counts = np.array(processed_data_counts)

#                 denom += (denom == 0).astype(int)
#                 prev_weight = processed_data_counts / denom
#                 curr_weight = group_count / denom

#                 res[group_wise_key] \
#                     = (prev_weight * res[group_wise_key] + curr_weight * group_acc)

#             res[key] += (correct).sum().item()

#     def prune_metrics_mappings(original_metrics_mappings, keys_to_prune):
#         metrics_mappings = []
#         for metric_tuple in original_metrics_mappings:
#             metric_name = metric_tuple[0]
#             if metric_name not in keys_to_prune:
#                 metrics_mappings.append(metric_tuple)
#         return tuple(metrics_mappings)

#     def aggregate_over_submodels(res, submodel_values, suffix=''):

#         if len(submodel_values) > 0:

#             res["best_single_model" + suffix] = max(
#                 submodel_values
#             )
#             res["mean_single_model" + suffix] = np.array(submodel_values).mean()

#     # Ensure the model is in evaluation mode
#     model.to(device)
#     model.eval()
#     if feature_extractor is not None:
#         feature_extractor.to(device)
#         feature_extractor.eval()

#     # correct = 0
#     total = 0

#     res = {}

#     # detailed results = per sample results
#     if return_detailed:
#         detailed_results = {}
#     else:
#         detailed_results = None

#     mappings_pruned = False
#     per_group_totals = {}

#     with torch.no_grad():  # No need to track gradients during evaluation
#         for batch_idx, data in enumerate(tqdm(dataloader)):

#             inputs = data[0]
#             labels = data[1]
#             if len(data) > 2:
#                 assert len(data) in [3, 4]
#                 metadata = data[2]
#             else:
#                 metadata = None

#             inputs, labels = inputs.to(device), labels.to(device)

#             if feature_extractor is not None:
#                 inputs = feature_extractor(inputs)
#             if hasattr(model, "soup") and model.soup is not None:
#                 # TODO(Alex | 13.05.2024): put it inside method model.forward_soup
#                 if hasattr(model, "feature_extractor") and model.feature_extractor is not None:
#                     soup_inputs = model.apply_feature_extractor(inputs)
#                 else:
#                     soup_inputs = inputs
#                 soup_output = model.soup(soup_inputs)
#                 update_correct(
#                     res,
#                     "soup",
#                     soup_output,
#                     labels,
#                     detailed_results,
#                     metadata,
#                     per_group_totals
#                 )

#             outputs = model(inputs)

#             if isinstance(outputs, list):

#                 assert model.weights is not None, \
#                     "Expect ensemble ensemble prediction mode"
#                 outputs = [output[1] for output in outputs]
#                 for i, output in enumerate(outputs):
#                     if i == len(outputs) - 1:
#                         key = f"ensemble"
#                     else:
#                         key = f"submodel_{i}"
#                     update_correct(
#                         res,
#                         key,
#                         output,
#                         labels,
#                         detailed_results,
#                         metadata,
#                         per_group_totals
#                     )

#                 submodels_outputs = outputs[:-1]

#                 if metrics_mappings is not None:

#                     # to avoid OOM
#                     if prune_metrics and len(submodels_outputs) > 2 and not mappings_pruned:
#                         metrics_mappings = prune_metrics_mappings(
#                             metrics_mappings,
#                             prune_metrics
#                         )
#                         mappings_pruned = True

#                     record_diversity(
#                         res,
#                         submodels_outputs,
#                         torch.stack(submodels_outputs, dim=0),
#                         metrics_mappings,
#                         labels=labels,
#                         detailed_results=detailed_results
#                     )

#             else:
#                 update_correct(
#                     res,
#                     "single_model",
#                     outputs,
#                     labels,
#                     detailed_results,
#                     metadata,
#                     per_group_totals
#                 )

#             total += labels.size(0)

#     keys_to_pop = []

#     res_extension_dict = {}

#     for key in res:

#         if ADJUSTED_GROUPWISE_KEY in key:
#             for group_id, value in enumerate(res[key]):
#                 res_extension_dict[key + f"_group_{group_id}"] = value
#             keys_to_pop.append(key)
#             continue

#         if (
#                 "ensemble" in key
#             or
#                 "submodel_" in key
#             or
#                 "single_model" == key
#             or
#                 "best_single_model" == key
#             or
#                 "soup" in key
#         ):
#             if "group" in key:
#                 divide_by = per_group_totals[key]
#             else:
#                 divide_by = total
#         else:
#             divide_by = len(dataloader)

#         if divide_by == 0:
#             assert "group" in key
#             keys_to_pop.append(key)
#         else:
#             res[key] /= divide_by

#     for key in keys_to_pop:
#         res.pop(key)

#     res |= res_extension_dict

#     # aggregate to worst groups
#     if len(per_group_totals) > 0:
#         tmp = {}
#         for key in res:

#             if "group" in key:
#                 original_key = key.split("_group")[0]
#                 if original_key not in tmp:
#                     tmp[original_key] = []
#                 tmp[original_key].append(res[key])

#         for key, value in tmp.items():
#             res[key + "_worst_group"] = min(value)

#     # aggregate to best and mean model
#     if len(res) == 1:
#         res = res["single_model"]
#     else:

#         aggregate_over_submodels(
#             res,
#             [
#                 value for key, value in res.items()
#                     if "submodel" in key and not "group" in key
#             ]
#         )
#         aggregate_over_submodels(
#             res,
#             [
#                 value for key, value in res.items()
#                     if "submodel" in key and "worst_group" in key
#             ],
#             suffix="_worst_group"
#         )

#     if return_detailed:
#         return res, detailed_results

#     return res


# def evaluate_ensemble(
#     ensemble,
#     dataloader,
#     device=torch.device("cuda:0"),
#     feature_extractor=None,
#     metrics_mappings=None,
#     return_detailed=False,
#     prune_metrics=["dis"],
#     average_after_softmax=False,
#     evaluation_func=None,
#     evaluation_kwargs={},
#     logger=None
# ):

#     prev_feature_extractor = ensemble.feature_extractor

#     if feature_extractor is not None:
#         ensemble.feature_extractor = feature_extractor
#         ensemble.feature_extractor.to(device)

#     ensemble.to(device)
#     num_submodels = len(ensemble.submodels)

#     prev_weights = ensemble.weights
#     ensemble.set_weights([1.0 for _ in range(num_submodels)], normalize=False)
#     if average_after_softmax:
#         prev_softmax_ensemble = ensemble.softmax_ensemble
#         ensemble.softmax_ensemble = True

#     if evaluation_func is None:
#         res = evaluate_model(
#             ensemble,
#             dataloader,
#             device,
#             metrics_mappings=metrics_mappings,
#             return_detailed=return_detailed,
#             prune_metrics=prune_metrics,
#             **evaluation_kwargs
#         )
#     else:
#         res = evaluation_func(
#             ensemble,
#             dataloader,
#             device,
#             **evaluation_kwargs
#         )
#     if average_after_softmax:
#         ensemble.softmax_ensemble = prev_softmax_ensemble
#     ensemble.set_weights(prev_weights)
#     if feature_extractor is not None:
#         ensemble.feature_extractor.to(torch.device("cpu"))
#     ensemble.feature_extractor = prev_feature_extractor
#     ensemble.to("cpu")

#     return res


# def apply_pairwise(iterable, func):

#     if len(iterable) == 1:
#         return iterable

#     pairs = itertools.combinations(iterable, 2)
#     res = []
#     for a, b in pairs:
#         res.append(func(a, b))
#     return res


# def get_validation_dataloaders(
#     train_batch_size,
#     eval_batch_size,
#     easy_robust_config,
#     num_workers,
#     eval_transform,
#     logger,
#     get_val_dataloader=??
# ):
#     dataset_types = get_with_assert(easy_robust_config, "dataset_types")

#     val_dataloaders = {}

#     if eval_transform is None:
#         eval_transform = make_default_test_transforms_imagenet()

#     for dataset_type in dataset_types:
#         assert dataset_type not in val_dataloaders, "Duplicate dataset type"
#         if dataset_type in ["imagenet_a", "imagenet_r", "imagenet_v2"]:
#             val_dataloaders[dataset_type] = get_imagenet_arv2_dataloader(
#                 train_batch_size=train_batch_size,
#                 eval_batch_size=eval_batch_size,
#                 easyrobust_config=easy_robust_config,
#                 num_workers=num_workers,
#                 eval_transform=eval_transform,
#                 logger=logger,
#                 dataset_type=dataset_type
#             )
#         elif dataset_type == "imagenet_hard":
#             val_dataloaders[dataset_type] = get_imagenet_hard_dataloader(
#                 train_batch_size=train_batch_size,
#                 eval_batch_size=eval_batch_size,
#                 easyrobust_config=easy_robust_config,
#                 num_workers=num_workers,
#                 eval_transform=eval_transform,
#                 logger=logger
#             )
#         elif dataset_type == "imagenet_c":
#             val_dataloaders |= get_imagenet_c_dataloader(
#                 train_batch_size,
#                 eval_batch_size,
#                 easy_robust_config,
#                 num_workers,
#                 eval_transform,
#                 logger
#             )
#         elif dataset_type == "openimages":
#             val_dataloaders[dataset_type] = get_openimages_dataloader(
#                 train_batch_size,
#                 eval_batch_size,
#                 easy_robust_config,
#                 num_workers,
#                 eval_transform,
#                 logger
#             )
#         elif dataset_type == "from_folder":
#             val_dataloaders |= get_from_folder_dataloader(
#                 train_batch_size,
#                 eval_batch_size,
#                 easy_robust_config,
#                 num_workers,
#                 eval_transform,
#                 logger
#             )
#         elif dataset_type == "imagenet_d":
#             val_dataloaders |= get_imagenet_d_dataloaders(
#                 train_batch_size,
#                 eval_batch_size,
#                 easy_robust_config,
#                 num_workers,
#                 eval_transform,
#                 logger
#             )
#         else:
#             raise_unknown(
#                 "dataset type",
#                 dataset_type,
#                 "easy_robust_config"
#             )
#     return None, val_dataloaders


def ood_detection_only_warning(logger):
    # logger.error(
    #     "All images have label 0 here. "
    #     "This dataloader is purely for OOD detection."
    # )
    error_or_print(
        "All images have label 0 here. "
        "This dataloader is purely for OOD detection.",
        logger
    )


def extract_tar(tar_path, folder):
    run_cmd_through_popen(
        f"tar -zvxf {tar_path} -C {folder}",
        # verbose=True,
        logger=None
    )


def download_file(file_path, download_url):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if "google" in download_url:
            download_type = "gdrive"
        else:
            download_type = "wget"

        if download_type == "wget":
            run_cmd_through_popen(
                f"wget {download_url} -O {file_path}",
                # verbose=True,
                logger=None
            )
        else:
            assert download_type == "gdrive"
            gdown.download(
                download_url,
                file_path,
                quiet=False,
                use_cookies=False,
                fuzzy=True
            )
            # gdown


def download_and_extract_tar(data_dir, download_url, name=None):
    # def make_extract_cmd(file):
    #     return (
    #         f"tar -zxf {file} " # -C $FOLDER
    #         # f"&& rm {file}"
    #     )
    if name is None:
        cur_time = str(get_current_time()).replace(' ', '_')
        name = f"tmp_tar_{cur_time}"
    parent_folder = os.path.dirname(data_dir)
    downloaded_tar = os.path.join(parent_folder, f"{name}.tar.gz")
    download_file(downloaded_tar, download_url)
    # if not os.path.exists(downloaded_tar):
    #     os.makedirs(os.path.dirname(downloaded_tar), exist_ok=True)
    #     if "google" in download_url:
    #         download_type = "gdrive"
    #     else:
    #         download_type = "wget"

    #     if download_type == "wget":
    #         run_cmd_through_popen(
    #             f"wget {download_url} -O {downloaded_tar}",
    #             # verbose=True,
    #             logger=None
    #         )
    #     else:
    #         assert download_type == "gdrive"
    #         gdown.download(
    #             download_url,
    #             downloaded_tar,
    #             quiet=False,
    #             use_cookies=False
    #         )
    #         # gdown

    # downloaded_tar = None # ??
    # gdown(test)
    # gdown list
    extract_tar(downloaded_tar, parent_folder)
    remove_file_or_folder(downloaded_tar)



