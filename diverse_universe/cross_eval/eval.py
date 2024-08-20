import os
import sys
import torch
import shutil
import sklearn
# import sklearn.metrics as skmetrics
import numpy as np
# import torch.nn.functional as F
from tqdm import tqdm
# import pandas as pd
# from IPython.display import display
from stuned.utility.utils import (
    raise_unknown,
    get_with_assert,
    pretty_json,
    log_or_print,
    append_dict,
    # aggregate_tensors_by_func,
    # apply_pairwise
    # check_dict,
    # update_dict_by_nested_key,
)
from stuned.utility.logger import (
    # LOGGING_CONFIG_KEY,
    # GDRIVE_FOLDER_KEY,
    make_logger,
    try_to_log_in_csv
)
# from stuned.local_datasets.imagenet1k import (
#     IMAGENET_TRAIN_NUM_SAMPLES
# )
from stuned.utility.configs import (
    RUN_PATH_CONFIG_KEY
)


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
# from stuned.utility.utils import (
#     raise_unknown,
#     # check_dict,
#     get_with_assert,
#     # update_dict_by_nested_key,
#     pretty_json
# )
# from stuned.utility.logger import (
#     LOGGING_CONFIG_KEY,
#     GDRIVE_FOLDER_KEY,
#     make_logger,
#     try_to_log_in_csv
# )
from diverse_universe.train.losses import (
    # PER_SAMPLE_METRIC_NAMES,
    div_different_preds,
    div_different_preds_per_sample,
    div_continous_unique_per_sample,
    div_continous_unique,
    div_var,
    get_probs,
    record_diversity,
    ens_entropy,
    average_entropy,
    mutual_information,
    average_energy,
    average_max_logit,
    ens_entropy_per_sample,
    average_entropy_per_sample,
    mutual_information_per_sample,
    average_energy_per_sample,
    average_max_logit_per_sample,
    a2d_score,
    a2d_score_per_sample
    # METRICS,
    # div_ortega,
    # div_mean_logit,
    # div_max_logit,
    # div_mean_prob,
    # div_max_prob,
    # div_entropy,
    # div_different_preds_per_model,
    # focal_modifier,
)
# from diverse_universe.utility.for_notebooks import (
#     # make_cross_dict,
#     # make_ensembles_from_paths,
#     # plot_table_from_cross_dict,
#     # get_long_table,
#     # format_number,
#     # make_models_dict_from_huge_string,
#     # evaluate_ensembles,
#     # ensemble_from_multiple_paths,
#     # evaluate_ensemble
# )
from diverse_universe.local_models.ensemble import (
    is_ensemble,
    get_ensemble,
    make_ensembles
    # REDNECK_ENSEMBLE_KEY,
    # SINGLE_MODEL_KEY,
    # POE_KEY,
    # make_redneck_ensemble,
    # split_linear_layer
)
# from diverse_universe.local_models.wrappers import (
#     wrap_model
# )
from diverse_universe.train.losses import (
    get_metrics_mapping
)
# from diverse_universe.local_models.ensemble import (
#     make_ensembles_from_paths
# )
from diverse_universe.local_models.common import (
    wrap_model,
    make_models_dict_from_huge_string
)
from diverse_universe.local_datasets.common import (
    # get_dataloaders,
    make_dataloaders,
    # make_hdf5_dataloader_from_path,
    make_cached_dataloaders
)
# from stuned.local_datasets.imagenet1k import (
#     IMAGENET_TRAIN_NUM_SAMPLES
# )
from diverse_universe.local_datasets.wilds import (
    metadata_to_group
)
# from utility.configs import (
#     ANY_KEY,
#     find_nested_keys_by_keyword_in_config,
#     normalize_paths
# )
# from train_eval.losses import (
#     ORTHOGONAL_GRADIENTS_LOSS,
#     ON_MANIFOLD_LOSS,
#     DIV_VIT_LOSS,
#     TASK_LOSS_WEIGHT_KEY,
#     GRADIENT_BASED_LOSSES_WEIGHTS_KEY,
#     GRADIENT_BASED_LOSSES_KEY,
#     DIVERSE_GRADIENTS_LOSS_KEY,
#     DIVDIS_LOSS_KEY,
#     LAMBDA_KEY
# )
# from train_eval.models import (
#     REDNECK_ENSEMBLE_KEY,
#     SINGLE_MODEL_KEY,
#     POE_KEY
# )
# from utility.imports import (
#     FROM_CLASS_KEY
# )
# from local_algorithms.div_dis import (
#     DIV_DIS_MODEL_NAME,
#     DIV_DIS_ALGO_NAME,
#     CLASSIFICATION_LOSS_WEIGHT_KEY
# )
# from local_datasets.common import (
#     UNLABELED_DATASET_KEY
# )
# from local_datasets.tiny_imagenet import (
#     TINY_IMAGENET,
#     TINY_IMAGENET_TEST
# )
# from local_datasets.from_h5 import (
#     FROM_H5
# )
# from local_algorithms.div_dis import (
#     DIVDIS_DIVERSITY_WEIGHT_KEY
# )
# from local_datasets.imagenet1k import (
#     IMAGENET_KEY
# )
# from stuned.utility.configs import (
#     RUN_PATH_CONFIG_KEY
# )
# from diverse_universe.local_datasets.easy_robust import (
#     IN_C_DATALOADERS_NAMES,
#     extract_in_c_paths
# )
# from diverse_universe.local_datasets.imagenet_c import (
#     IN_C_DATALOADERS_NAMES,
#     extract_in_c_paths
# )
sys.path.pop(0)


# # TODO(Alex | 11.05.2024): move to utils
# SCRATCH_LOCAL = os.path.join(
#     os.sep,
#     "scratch_local",
#     f"{os.environ.get('USER')}-{os.environ.get('SLURM_JOB_ID')}",
# )
ADJUSTED_GROUPWISE_KEY = "_adjusted"
# DEFAULT_ROUND = 9
# HYPERPARAM_PREFIX = "__hyperparam__"
# IN_VAL_CACHED_2LAYER_PATH = \
#     "/mnt/qb/work/oh/arubinstein17/cache/ImageNet1k/val_cache/ed4f10f4ddeb07f1d876_torch_load_block_-1_model_imagenet1k_dataset_1_epochs_50000_samples.hdf5"
EXTENDED_METRICS = [
    tuple(["ens_entropy", ens_entropy]),
    tuple(["average_entropy", average_entropy]),
    tuple(["mutual_information", mutual_information]),
    tuple(["average_energy", average_energy]),
    tuple(["average_max_logit", average_max_logit]),
    tuple(["a2d_score", a2d_score])
]
EXTENDED_METRICS_PER_SAMPLE = [
    tuple(["ens_entropy_per_sample", ens_entropy_per_sample]),
    tuple(["average_entropy_per_sample", average_entropy_per_sample]),
    tuple(["mutual_information_per_sample", mutual_information_per_sample]),
    tuple(["average_energy_per_sample", average_energy_per_sample]),
    tuple(["average_max_logit_per_sample", average_max_logit_per_sample]),
    tuple(["a2d_score_per_sample", a2d_score_per_sample])
]
CONF = "conf"


def patch_eval_config(experiment_config):
    pass


def check_eval_config(experiment_config, config_path, logger=make_logger()):
    pass


def cross_eval(
    experiment_config,
    logger=make_logger(),
    processes_to_kill_before_exiting=[]
):

    models_dict, model_to_prop_dict = prepare_models(
        get_with_assert(experiment_config, "models")
    )
    eval_type = get_with_assert(experiment_config, "eval_type")

    exp_type = get_with_assert(experiment_config, "experiment_type")
    dataloaders_dict = prepare_dataloaders(
        exp_type,
        eval_type,
        cached_datasets_info=get_with_assert(
            experiment_config,
            "cached_datasets_info"
        )
    )

    local_results_save_path = os.path.join(
        experiment_config[RUN_PATH_CONFIG_KEY],
        "results.pkl"
    )
    results_save_path = experiment_config.get(
        "results_save_path"
    )
    if results_save_path is None:
        results_save_path = local_results_save_path

    ood_metric_type = experiment_config.get("ood_metric_type", "default")
    assert ood_metric_type in [
        "default",
        # "minimal",
        # "rebuttal_extension",
        # "a2d_score"
    ]

    metrics_mappings = make_metrics_mappings(eval_type, ood_metric_type)

    if eval_type == "ood_gen":

        eval_kwargs = {"cook_soup": True}

        if exp_type == "waterbirds_dataloaders":
            eval_kwargs["metadata_to_group"] = metadata_to_group

        res = make_cross_dict(
            models_dict=models_dict,
            dataloaders_dict=dataloaders_dict,
            save_path=results_save_path,
            device=torch.device("cuda:0"),
            feature_extractor=None,
            wrap=None,
            verbose=False,
            metrics_mappings=metrics_mappings
            # [
            #     ("div_different_preds", div_different_preds), # TODO(Alex 06.05.2024): specify by config
            #     ("div_continous_unique", div_continous_unique),
            #     ("var", div_var)
            # ]
            ,
            prune_metrics=[],
            evaluation_kwargs=eval_kwargs,
            logger=logger,
            model_to_prop_dict=model_to_prop_dict
        )
    elif eval_type == "ood_det":

        if exp_type in ["deit3b_dataloaders", "debug"]:
            id_dataloader = dataloaders_dict["in_val"]
        else:
            assert exp_type == "waterbirds_dataloaders"
            id_dataloader = dataloaders_dict["train_divdis"]
        res = make_cross_dict(
            models_dict=models_dict,
            dataloaders_dict=dataloaders_dict,
            save_path=results_save_path,
            device=torch.device("cuda:0"),
            evaluation_func=evaluate_ood_detection,
            evaluation_kwargs={
                "metrics_mappings": metrics_mappings
                # [
                #     ("div_different_preds_per_sample", div_different_preds_per_sample),
                #     ("div_continous_unique_per_sample", div_continous_unique_per_sample)
                # ]
                ,
                "cache": {},
                "id_dataloader": id_dataloader # TODO(Alex | 06.05.2024): specify by config
            },
            model_to_prop_dict=model_to_prop_dict,
            logger=logger
        )
    else:
        raise NotImplementedError()
    if not os.path.exists(local_results_save_path):
        shutil.copyfile(results_save_path, local_results_save_path)
    logger.log(pretty_json(res))
    try_to_log_in_csv(logger, "results_save_path", results_save_path)


def prepare_models(models_config):
    check_for_duplicates(models_config)
    models_dict = {}
    model_to_prop_dict = {}
    # for model_group_config in models_config.values():
    for model_group_key, model_group_config in models_config.items():
        if not isinstance(model_group_config, dict):
            assert model_group_key == "total_groups"
            continue
        model_group_dict, model_to_prop_group_dict = \
            prepare_model_group(model_group_config)
        models_dict |= model_group_dict
        model_to_prop_dict |= model_to_prop_group_dict
    return models_dict, model_to_prop_dict


def prepare_model_group(model_group_config):
    fields = get_with_assert(model_group_config, "fields")
    model_paths = get_with_assert(model_group_config, "model_paths")
    model_group_prefix = get_with_assert(
        model_group_config,
        "model_group_prefix"
    )
    models, model_name_to_prop = make_models_dict_from_huge_string(
        model_paths,
        fields,
        id_prefix=model_group_prefix
    )
    return models, model_name_to_prop


def prepare_dataloaders(exp_type, eval_type, cached_datasets_info):
    if exp_type == "deit3b_dataloaders":
        return prepare_deit3b_dataloaders(eval_type, cached_datasets_info)
    # elif exp_type == "waterbirds_dataloaders":
    #     return prepare_waterbirds_dataloaders(eval_type)
    # elif exp_type == "debug":
    #     return prepare_debug_dataloaders(cached_datasets_info)
    else:
        raise_unknown(exp_type, "experiment_type", "prepare_dataloaders")


# def prepare_debug_dataloaders():
#     # TODO(Alex | 23.07.2024): Move paths to config
#     # ??
#     cached_dataloaders_deit3b_dict = make_cached_dataloaders(
#         {
#             "in_val": IN_VAL_CACHED_2LAYER_PATH,
#             "imagenet_a": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/imagenet_a/f37703ce0aea0e55ab2c_torch_load_block_-1_model_imagenet_a_dataset_1_epochs_7500_samples.hdf5",
#             "imagenet_r": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/imagenet_r/c9fe46d6544688f46907_torch_load_block_-1_model_imagenet_r_dataset_1_epochs_30000_samples.hdf5"
#         }
#     )
#     deit3b_2layers_all_dataloaders_dict = {
#         "imagenet_a": (cached_dataloaders_deit3b_dict["imagenet_a"], "ina"),
#         "imagenet_r": (cached_dataloaders_deit3b_dict["imagenet_r"], "inr"),
#         "in_val": cached_dataloaders_deit3b_dict["in_val"]
#     }
#     return deit3b_2layers_all_dataloaders_dict


def subset_dict_by_keys(input_dict, keys):
    return {
        key: input_dict[key] for key in keys
    }


# def prepare_waterbirds_dataloaders(eval_type, batch_size=128, num_workers=4):

#     waterbirds_config_divdis = {
#         "type": "waterbirds",
#         'waterbirds': {
#             "root_dir": None,
#             "keep_metadata": (eval_type == "ood_gen"),
#             "use_waterbirds_divdis": True
#         }
#     }

#     waterbirds_train_dataloader_divdis, waterbirds_val_dataloaders_divdis, _ = make_dataloaders(
#         waterbirds_config_divdis,
#         train_batch_size=batch_size,
#         eval_batch_size=batch_size,
#         num_workers=num_workers,
#         to_train=True
#     )
#     waterbirds_val_dataloaders_divdis["train"] = waterbirds_train_dataloader_divdis

#     if eval_type == "ood_gen":

#         waterbirds_config = {
#             "type": "waterbirds",
#             'waterbirds': {
#                 "root_dir": "/mnt/qb/work/oh/arubinstein17/cache/wilds/waterbirds",
#                 # "root_dir": SCRATCH_LOCAL,
#                 # "root_dir": "./",
#                 "keep_metadata": True,
#                 # "eval_transforms": "ImageNetEval"
#             }
#         }

#         waterbirds_train_dataloader, waterbirds_val_dataloaders, _ = make_dataloaders(
#             waterbirds_config,
#             train_batch_size=batch_size,
#             eval_batch_size=batch_size,
#             num_workers=num_workers,
#             to_train=True
#         )
#         waterbirds_val_dataloaders["train"] = waterbirds_train_dataloader

#         waterbirds_config_with_transforms = {
#             "type": "waterbirds",
#             'waterbirds': {
#                 "root_dir": "/mnt/qb/work/oh/arubinstein17/cache/wilds/waterbirds",
#                 # "root_dir": SCRATCH_LOCAL,
#                 # "root_dir": "./",
#                 "keep_metadata": True,
#                 "eval_transforms": "ImageNetEval"
#             }
#         }

#         waterbirds_train_dataloader_with_transforms, waterbirds_val_dataloaders_with_transforms, _ = make_dataloaders(
#             waterbirds_config_with_transforms,
#             train_batch_size=batch_size,
#             eval_batch_size=batch_size,
#             num_workers=num_workers,
#             to_train=True
#         )

#         waterbirds_val_dataloaders_with_transforms["train"] = waterbirds_train_dataloader_with_transforms

#         all_waterbirds_dataloaders_dict = (
#                 waterbirds_val_dataloaders
#             |
#                 update_key_names(
#                     waterbirds_val_dataloaders_with_transforms,
#                     "_with_transform"
#                 )
#         )
#     else:
#         assert eval_type == "ood_det"
#         all_waterbirds_dataloaders_dict = {}

#     all_waterbirds_dataloaders_dict |= update_key_names(
#         waterbirds_val_dataloaders_divdis,
#         "_divdis"
#     )

#     return all_waterbirds_dataloaders_dict


def update_key_names(d, suffix):
    return {
        key + suffix: value
            for key, value
                in d.items()
    }


def prepare_deit3b_dataloaders(eval_type, cached_datasets_info):

    # assert exp_type == "deit3b"
    # Deit3B

    # # TODO(Alex | 23.07.2024): Move paths to config
    # # in_c_dataloaders = extract_in_c_paths(
    # in_c_dataset_paths = extract_in_c_paths(
    #     base_path=get_with_assert(
    #         cached_datasets_info,
    #         "base_path"
    #     ),
    #     strengths=cached_datasets_info.get("in_c_strengths", [1, 5]),
    #     # filename_substring="torch_load_block_-1_model"
    #     filename_substring=get_with_assert(
    #         cached_datasets_info,
    #         "filename_substring"
    #     )
    # )

    # path_mapping = get_with_assert(cached_datasets_info, "path_mapping")

    # in_val_5k_hard = make_hdf5_dataloader_from_path(
    #     in_val_cached_2layer_path,
    #     32,
    #     0,
    #     indices_to_keep_path="/mnt/qb/work/oh/arubinstein17/cache/ImageNet1k/indices/least_conf_5000_29af17cd83868879248c_val.pt",
    #     reverse_indices=False
    # )

    # in_val_45k_easy = make_hdf5_dataloader_from_path(
    #     in_val_cached_2layer_path,
    #     32,
    #     0,
    #     indices_to_keep_path="/mnt/qb/work/oh/arubinstein17/cache/ImageNet1k/indices/least_conf_5000_29af17cd83868879248c_val.pt",
    #     reverse_indices=True
    # )

    # in_train_3068_hard = make_hdf5_dataloader_from_path(
    #     "/mnt/qb/work/oh/arubinstein17/cache/ImageNet1k/train_cache/ed4f10f4ddeb07f1d876_torch_load_block_-1_model_imagenet1k_dataset_4_epochs_1281167_samples.hdf5",
    #     32,
    #     0,
    #     indices_to_keep_path="/mnt/qb/work/oh/arubinstein17/cache/ImageNet1k/indices/least_conf_3068_29af17cd83868879248c_train.pt",
    #     reverse_indices=False,
    #     max_chunk_size=IMAGENET_TRAIN_NUM_SAMPLES,
    #     total_samples=IMAGENET_TRAIN_NUM_SAMPLES
    # )

    # cached_datasets:
    # ood_det_keys: ??
    # ood_gen_keys: ??
    # path_mapping: ??
    # base_path: ??
    # filename_substring: ??
    # ?? read from dict in code

    #     cached_datasets_info ??

    # cached_dataloaders_deit3b_dict = make_cached_dataloaders(
    #     {
    #         # "in_val": IN_VAL_CACHED_2LAYER_PATH,
    #         "in_val": get_with_assert(path_mapping, "in_val"),
    #     } | in_c_dataloaders
    # )

    # in_c_dataloaders_dict = subset_dict_by_keys(
    #     cached_dataloaders_deit3b_dict,
    #     in_c_dataloaders.keys()
    # )

    # deit3b_2layers_all_dataloaders_dict = {
    #     "in_val": cached_dataloaders_deit3b_dict["in_val"]
    # }

    # # TODO(Alex | 23.07.2024): Move paths to config
    # cached_dataloaders_deit3b_dict = {} # tmp
    # cached_dataloaders_deit3b_dict |= make_cached_dataloaders(
    #     {
    #         "iNaturalist": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/iNaturalist/214988c198612f2682e6_torch_load_block_-1_model_iNaturalist_dataset_1_epochs_10000_samples.hdf5",
    #         # "iNaturalist": get_with_assert(path_mapping, "iNaturalist"),
    #         "OpenImages": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/openimages/816f6e9e7fa686d093e8_torch_load_block_-1_model_openimages_dataset_1_epochs_17632_samples.hdf5"
    #     }
    # )



    # if eval_type == "ood_gen": # ?? remove

    #     # # TODO(Alex | 23.07.2024): Move paths to config
    #     # in_d_dataloaders = {
    #     #     "in_d_background": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/background/656ee9569bfbcf377ea9_torch_load_block_-1_model_background_dataset_1_epochs_3764_samples.hdf5",
    #     #     "in_d_texture": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/texture/8ecfa96f9135692c2cf2_torch_load_block_-1_model_texture_dataset_1_epochs_498_samples.hdf5",
    #     #     "in_d_material": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/material/116c69294993da7f1b7f_torch_load_block_-1_model_material_dataset_1_epochs_573_samples.hdf5"
    #     # }

    #     # TODO(Alex | 23.07.2024): Move paths to config
    #     cached_dataloaders_deit3b_dict |= make_cached_dataloaders(
    #         {
    #         "imagenet_a": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/imagenet_a/f37703ce0aea0e55ab2c_torch_load_block_-1_model_imagenet_a_dataset_1_epochs_7500_samples.hdf5",
    #         "imagenet_r": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/imagenet_r/c9fe46d6544688f46907_torch_load_block_-1_model_imagenet_r_dataset_1_epochs_30000_samples.hdf5",
    #         "mixed_rand": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/mixed_rand/bf802cc68d4a8b524c38_torch_load_block_-1_model_mixed_rand_dataset_1_epochs_4050_samples.hdf5",
    #         "mixed_same": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/mixed_same/03839176c08efd3bf9e0_torch_load_block_-1_model_mixed_same_dataset_1_epochs_4050_samples.hdf5",
    #         "only_bg_t": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/only_bg_t/82c8996731b022fe6a5f_torch_load_block_-1_model_only_bg_t_dataset_1_epochs_4050_samples.hdf5",
    #         "no_fg": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/no_fg/5cc8e5036cec4c12a397_torch_load_block_-1_model_no_fg_dataset_1_epochs_4050_samples.hdf5",
    #         "mixed_next": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/mixed_next/0cbfd71928f9d1f69cd2_torch_load_block_-1_model_mixed_next_dataset_1_epochs_4050_samples.hdf5",
    #         "only_fg": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/only_fg/3de26782660a000fb937_torch_load_block_-1_model_only_fg_dataset_1_epochs_4050_samples.hdf5",
    #         "original": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/original/d5967a112b317104ccb5_torch_load_block_-1_model_original_dataset_1_epochs_4050_samples.hdf5",
    #         "only_bg_b": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/only_bg_b/b2005edd362a19e1ae41_torch_load_block_-1_model_only_bg_b_dataset_1_epochs_4050_samples.hdf5",
    #         "stylized": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/stylized/6a559f1b1becae879c79_torch_load_block_-1_model_stylized_dataset_1_epochs_800_samples.hdf5",
    #         "sketch": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/sketch/ff0161b6b924453b0c68_torch_load_block_-1_model_sketch_dataset_1_epochs_800_samples.hdf5",
    #         "edge": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/edge/7f7ea165dfb604e47f3c_torch_load_block_-1_model_edge_dataset_1_epochs_160_samples.hdf5",
    #         "silhouette": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/silhouette/9def5ec3aecf90d0515c_torch_load_block_-1_model_silhouette_dataset_1_epochs_160_samples.hdf5",
    #         "cue-conflict": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/cue-conflict/5f1aa995aa7afccf46e8_torch_load_block_-1_model_cue-conflict_dataset_1_epochs_1280_samples.hdf5",
    #         # "in_val": in_val_cached_2layer_path,
    #         }
    #         # | in_d_dataloaders
    #     )

    #     # for dataset_key in get_with_assert(cached_datasets_info, "ood_gen_keys"):
    #     #     dataset_entry = get_with_assert(cached_datasets_info, dataset_key)
    #     #     # if isinstance(dataset_entry, tuple):
    #     #     #     dataset_path, wrapper_name = dataset_entry
    #     #     # else:
    #     #     #     dataset_path, wrapper_name = dataset_entry, None
    #     #     deit3b_2layers_all_dataloaders_dict[dataset_key] = dataset_entry

    #     deit3b_2layers_all_dataloaders_dict |= {
    #         "imagenet_a": (cached_dataloaders_deit3b_dict["imagenet_a"], "ina"),
    #         "imagenet_r": (cached_dataloaders_deit3b_dict["imagenet_r"], "inr"),
    #         "mixed_rand": (cached_dataloaders_deit3b_dict["mixed_rand"], "IN9"),
    #         "mixed_next": (cached_dataloaders_deit3b_dict["mixed_next"], "IN9"),
    #         "cue-conflict": (cached_dataloaders_deit3b_dict["cue-conflict"], "mvh"),
    #         "stylized": (cached_dataloaders_deit3b_dict["stylized"], "mvh"),
    #         "iNaturalist": cached_dataloaders_deit3b_dict["iNaturalist"],
    #         "OpenImages": cached_dataloaders_deit3b_dict["OpenImages"]
    #         # "in_val": cached_dataloaders_deit3b_dict["in_val"],
    #         # "in_val_5k_hard": in_val_5k_hard,
    #         # "in_val_45k_easy": in_val_45k_easy,
    #         # "in_train_3068_hard": in_train_3068_hard,
    #         # "fog_1": cached_dataloaders_deit3b_dict["fog_1"],
    #         # "fog_5": cached_dataloaders_deit3b_dict["fog_5"],
    #         # "defocus_blur_1": cached_dataloaders_deit3b_dict["defocus_blur_1"],
    #         # "defocus_blur_5": cached_dataloaders_deit3b_dict["defocus_blur_5"],
    #     }

    #     # deit3b_2layers_all_dataloaders_dict |= {
    #     #     key: cached_dataloaders_deit3b_dict[key] for key in in_c_dataloaders.keys()
    #     # }

    #     # deit3b_2layers_all_dataloaders_dict |= in_c_dataloaders_dict

    #     # deit3b_2layers_all_dataloaders_dict |= subset_dict_by_keys(
    #     #     cached_dataloaders_deit3b_dict,
    #     #     in_d_dataloaders.keys()
    #     # )

    #     # deit3b_2layers_all_dataloaders_dict |= {
    #     #     key: cached_dataloaders_deit3b_dict[key] for key in in_d_dataloaders.keys()
    #     # }
    # else:
    #     assert eval_type == "ood_det"
    #     # cached_dataloaders_deit3b_dict |= make_cached_dataloaders(
    #     #     {
    #     #         "iNaturalist": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/iNaturalist/214988c198612f2682e6_torch_load_block_-1_model_iNaturalist_dataset_1_epochs_10000_samples.hdf5",
    #     #         "OpenImages": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/openimages/816f6e9e7fa686d093e8_torch_load_block_-1_model_openimages_dataset_1_epochs_17632_samples.hdf5"
    #     #     }
    #     # )
    #     deit3b_2layers_all_dataloaders_dict |= {
    #         # "imagenet_a": (cached_dataloaders_deit3b_dict["imagenet_a"], "ina"),
    #         # "imagenet_r": (cached_dataloaders_deit3b_dict["imagenet_r"], "inr"),
    #         # "mixed_rand": (cached_dataloaders_deit3b_dict["mixed_rand"], "IN9"),
    #         # "mixed_next": (cached_dataloaders_deit3b_dict["mixed_next"], "IN9"),
    #         # "cue-conflict": (cached_dataloaders_deit3b_dict["cue-conflict"], "mvh"),
    #         # "in_val": cached_dataloaders_deit3b_dict["in_val"],
    #         "iNaturalist": cached_dataloaders_deit3b_dict["iNaturalist"],
    #         "OpenImages": cached_dataloaders_deit3b_dict["OpenImages"]
    #         # "in_train_3068_hard": in_train_3068_hard
    #     }
    # deit3b_2layers_all_dataloaders_dict |= in_c_dataloaders_dict

    # path_mapping = get_with_assert(cached_datasets_info, "path_mapping")

        # dataset_names = get_with_assert(
        #     cached_datasets_info,
        #     f"{eval_type}_keys"
        # )
        # dataset_paths_dict |= in_c_dataset_paths
    # for dataset_key in get_with_assert(
    #     cached_datasets_info,
    #     f"{eval_type}_keys"
    # ):
    #     dataset_entry = get_with_assert(cached_datasets_info, dataset_key)
    #     # if isinstance(dataset_entry, list):
    #     #     dataset_path, wrapper_name = dataset_entry
    #     # else:
    #     #     dataset_path, wrapper_name = dataset_entry, None
    #     # in_c_dataset_paths ??
    #     # wrappers

    #     deit3b_2layers_all_dataloaders_dict[dataset_key] = dataset_entry

    # in_c_dataloaders_dict = subset_dict_by_keys(
    #     cached_dataloaders_deit3b_dict,
    #     get_with_assert(
    #         cached_datasets_info,
    #         f"{eval_type}_keys"
    #     )
    # )

    deit3b_2layers_all_dataloaders_dict = make_cached_dataloaders(
        subset_dict_by_keys(
            get_with_assert(cached_datasets_info, "path_mapping"),
            get_with_assert(cached_datasets_info, f"{eval_type}_keys")
        )
    )

    wrappers_dict = cached_datasets_info.get("wrappers", {})

    # deit3b_2layers_all_dataloaders_dict ??

    # if len(wr)
    # for wrapper_key
    for dataset_name in deit3b_2layers_all_dataloaders_dict.keys():
        if dataset_name in wrappers_dict:
            deit3b_2layers_all_dataloaders_dict[dataset_name] = (
                deit3b_2layers_all_dataloaders_dict[dataset_name],
                wrappers_dict[dataset_name]
            )

    return deit3b_2layers_all_dataloaders_dict


def joint_res(res_1, res_2, key):
    def add_single(true_labels, y_score, single_res, class_id):
        for value in single_res:
            true_labels.append(class_id)
            y_score.append(value)
    true_labels = []
    y_score = []
    add_single(true_labels, y_score, res_1[key], 0)
    add_single(true_labels, y_score, res_2[key], 1)
    return true_labels, y_score


# TODO(Alex | 23.07.2024): Make it more efficient and less embarrasing
# def compute_roc_auc(detailed_res_id, detailed_res_ood, verbose=True):

#     labels_div, divs = joint_res(
#         detailed_res_id,
#         detailed_res_ood, # class 1 has higher div
#         'div_different_preds_per_sample'
#     )

#     labels_cont_uniq, cont_uniq = joint_res(
#         detailed_res_id,
#         detailed_res_ood, # class 1 has higher div
#         'div_continous_unique_per_sample'
#     )

#     labels_conf, confs = joint_res(
#         detailed_res_ood["ensemble"],
#         detailed_res_id["ensemble"], # class 1 has higher conf
#         'conf'
#     )

#     # labels_conf_sm, confs_sm = joint_res(
#     #     ce_a2d_detailed_res_sm["ensemble"],
#     #     ce_a2d_detailed_res_in_val_sm["ensemble"], # class 1 has higher conf
#     #     'conf'
#     # )

#     labels_conf_0, confs_0 = joint_res(
#         detailed_res_ood["submodel_0"],
#         detailed_res_id["submodel_0"], # class 1 has higher conf
#         'conf'
#     )

#     # roc_auc_conf = sklearn.metrics.roc_auc_score(labels_conf, confs)
#     # # roc_auc_conf_sm = sklearn.metrics.roc_auc_score(labels_conf_sm, confs_sm)
#     # roc_auc_conf_0 = sklearn.metrics.roc_auc_score(labels_conf_0, confs_0)
#     # roc_auc_divs = sklearn.metrics.roc_auc_score(labels_div, divs)
#     # roc_auc_cont_uniq = sklearn.metrics.roc_auc_score(labels_cont_uniq, cont_uniq)
#     roc_auc_conf = skmetrics.roc_auc_score(labels_conf, confs)
#     # roc_auc_conf_sm = sklearn.metrics.roc_auc_score(labels_conf_sm, confs_sm)
#     roc_auc_conf_0 = skmetrics.roc_auc_score(labels_conf_0, confs_0)
#     roc_auc_divs = skmetrics.roc_auc_score(labels_div, divs)
#     roc_auc_cont_uniq = skmetrics.roc_auc_score(labels_cont_uniq, cont_uniq)


#     ##########
#     # TODO(Alex | 28.03.2024): remove duplication above and do it in a loop

#     res = {
#         "ensemble": roc_auc_conf,
#         "submodel_0": roc_auc_conf_0,
#         "divs": roc_auc_divs,
#         "cont_unique": roc_auc_cont_uniq
#     }

#     if verbose:
#         for metric_name, metric_value in res.items():
#             print(metric_name, metric_value)

#     return res


def compute_roc_auc(detailed_res_id, detailed_res_ood, metrics_mappings, verbose=True):

    def get_roc_auc_for_metric(
        detailed_res_id,
        detailed_res_ood,
        metric_name,
        reverse
    ):
        if reverse:
            labels, metric_values = joint_res(
                detailed_res_ood,
                detailed_res_id, # id has higher metric
                metric_name
            )
        else:
            labels, metric_values = joint_res(
                detailed_res_id,
                detailed_res_ood, # ood has higher metric
                metric_name
            )
        roc_auc = sklearn.metrics.roc_auc_score(labels, metric_values)
        return roc_auc

    def add_derivable_score(detailed_res):

        def invert_scores(scores):
            return [1 - score for score in scores]

        extension_dict = {}
        submodel_confs = []
        for key, value in detailed_res.items():
            if "ensemble" in key:
                extension_dict[key + "_" + CONF] = value[CONF]
            elif "submodel" in key:
                submodel_confs.append(value[CONF])
            elif (
                    "mutual_information_per_sample" in key
                or
                    "average_max_logit_per_sample" in key
            ):
                extension_dict["inv_" + key] = invert_scores(value)

        if len(submodel_confs) > 0:
            extension_dict["mean_submodel_" + CONF] \
                = np.mean(np.array(submodel_confs), axis=0)

        return detailed_res | extension_dict

    detailed_res_id = add_derivable_score(detailed_res_id)
    detailed_res_ood = add_derivable_score(detailed_res_ood)

    metric_names = [
        "ensemble_" + CONF,
        "mean_submodel_" + CONF,
        # 'div_different_preds_per_sample',
        # 'div_continous_unique_per_sample',
        # "ens_entropy_per_sample",
        # "average_entropy_per_sample",
        # "mutual_information_per_sample",
        # "average_energy_per_sample",
        # "average_max_logit_per_sample",
        # "inv_average_max_logit_per_sample",
        # "inv_mutual_information_per_sample",
    ] + [name for name, _ in metrics_mappings]

    if "mutual_information_per_sample" in metric_names:
        metric_names.append("inv_mutual_information_per_sample")

    if "average_max_logit_per_sample" in metric_names:
        metric_names.append("inv_average_max_logit_per_sample")

    res = {}
    # for metric_name in [
    #     "ensemble_" + CONF,
    #     "mean_submodel_" + CONF,
    #     # 'div_different_preds_per_sample',
    #     # 'div_continous_unique_per_sample',
    #     # "ens_entropy_per_sample",
    #     # "average_entropy_per_sample",
    #     # "mutual_information_per_sample",
    #     # "average_energy_per_sample",
    #     # "average_max_logit_per_sample",
    #     # "inv_average_max_logit_per_sample",
    #     # "inv_mutual_information_per_sample",
    # ] + metrics_mappings:
    for metric_name in metric_names:
        if (
                "conf" in metric_name
            or
                "average_max_logit_per_sample" in metric_name
            or
                "mutual_information_per_sample" in metric_name
        ):
            reverse = True
        else:
            reverse = False
        res[metric_name] = get_roc_auc_for_metric(
            detailed_res_id,
            detailed_res_ood,
            metric_name,
            reverse=reverse
        )

    if verbose:
        for metric_name, metric_value in res.items():
            print(metric_name, metric_value)

    return res


def evaluate_ood_detection(
    model,
    ood_dataloader,
    device,
    **evaluation_kwargs
):

    id_dataloader = get_with_assert(evaluation_kwargs, "id_dataloader")
    cache = get_with_assert(evaluation_kwargs, "cache")
    metrics_mappings = get_with_assert(evaluation_kwargs, "metrics_mappings")

    model_id = get_with_assert(evaluation_kwargs, "model_id")
    models_before_wrapper = get_with_assert(evaluation_kwargs, "models_before_wrapper")
    dataset_id = get_with_assert(evaluation_kwargs, "dataset_id")

    res_id = model_id + '_' + "id_dataloader"

    if res_id not in cache:

        print(f"Computing detailed result for {res_id}")

        assert len(models_before_wrapper) == 1
        original_model = models_before_wrapper[0]
        _, model_detailed_res_id = evaluate_ensemble(
            original_model,
            id_dataloader,
            device=device,
            feature_extractor=None,
            metrics_mappings=metrics_mappings,
            return_detailed=True,
            average_after_softmax=False
        )
        cache[res_id] = model_detailed_res_id

    model_detailed_res_id = cache[res_id]

    _, model_detailed_res_ood = evaluate_ensemble(
        model,
        ood_dataloader,
        device=device,
        feature_extractor=None,
        metrics_mappings=metrics_mappings,
        return_detailed=True,
        average_after_softmax=False
    )
    return compute_roc_auc(
        model_detailed_res_id,
        model_detailed_res_ood,
        verbose=False
    )


def evaluate_ensembles(
    ensembles,
    dataloader,
    device=torch.device("cuda:0"),
    feature_extractor=None,
    wrap=None,
    metrics_mappings=get_metrics_mapping(),
    prune_metrics=["dis"],
    evaluation_func=None,
    evaluation_kwargs={},
    logger=None
):

    if isinstance(ensembles[0], str):
        ensemble_paths = ensembles
        ensembles = [get_ensemble(ensemble_path) for ensemble_path in ensemble_paths]

    else:
        ensemble_paths = [f"ensemble_{i}" for i in range(len(ensembles))]

    evaluation_kwargs["models_before_wrapper"] = ensembles

    if evaluation_kwargs.get("cook_soup", False):
        for ensemble in ensembles:
            ensemble.cook_soup()

    if wrap is not None:
        ensembles = [wrap_model(ensemble, wrap) for ensemble in ensembles]

    res = {}
    for ensemble_path, ensemble in zip(ensemble_paths, ensembles):
        if is_ensemble(ensemble):

            res[ensemble_path] = evaluate_ensemble(
                ensemble,
                dataloader,
                device,
                feature_extractor,
                metrics_mappings=metrics_mappings,
                prune_metrics=prune_metrics,
                evaluation_func=evaluation_func,
                evaluation_kwargs=evaluation_kwargs,
                logger=logger
            )
        else:
            assert evaluation_func is None, \
                "can provide evaluation function only for ensembles"

            if feature_extractor is not None:
                log_or_print(
                    "Ignoring feature extractor for single model",
                    logger,
                    auto_newline=True
                )

            stat_value = evaluate_model(
                ensemble,
                dataloader,
                device,
                logger=logger
            )
            res[ensemble_path] = {
                "best_single_model": stat_value
            }

    return res


def evaluate_ensembles_on_dataloaders(
    ensembles,
    dataloaders,
    device=torch.device("cuda:0"),
    feature_extractor=None,
    wrap=None,
    verbose=True,
    metrics_mappings=get_metrics_mapping(),
    prune_metrics=["dis"],
    previous_results={},
    evaluation_func=None,
    evaluation_kwargs={},
    logger=None
):

    def print_using(new, default, name, what):
        log_or_print(
            f"Using {new} {what} "
            f"instead of {default} for dataloader {name}",
            logger,
            auto_newline=True
        )

    res = {}

    for name, dataloader in dataloaders.items():

        if name in previous_results:
            log_or_print(
                f"Skipping dataloader: {name}",
                logger,
                auto_newline=True
            )
            continue

        evaluation_kwargs["dataset_id"] = name
        per_dataloader_wrap = None
        per_dataloader_feature_extractor = "empty"
        if isinstance(dataloader, tuple):

            dataloader_tuple = dataloader
            dataloader, per_dataloader_wrap = dataloader_tuple[0], dataloader_tuple[1]
            print_using(per_dataloader_wrap, wrap, name, "wrapper")

            if len(dataloader_tuple) == 3:
                per_dataloader_feature_extractor = dataloader_tuple[2]
                print_using(
                    per_dataloader_feature_extractor,
                    type(feature_extractor),
                    name,
                    "feature_extractor"
                )
        log_or_print(
            f"evaluating on dataloader: {name}",
            logger,
            auto_newline=True
        )

        res[name] = evaluate_ensembles(
            ensembles,
            dataloader,
            device,
            (
                feature_extractor
                if per_dataloader_feature_extractor == "empty"
                else per_dataloader_feature_extractor
            ),
            wrap if per_dataloader_wrap is None else per_dataloader_wrap,
            metrics_mappings=metrics_mappings,
            prune_metrics=prune_metrics,
            evaluation_func=evaluation_func,
            evaluation_kwargs=evaluation_kwargs,
            logger=logger
        )

    if verbose:
        log_or_print(
            pretty_json(res),
            logger,
            auto_newline=True
        )
        log_or_print(
            pretty_json(aggregate_by_ensemble(res)),
            logger,
            auto_newline=True
        )
    return res



def aggregate_by_ensemble(eval_data):

    def aggregate_dicts(dicts):
        assert len(dicts)
        if isinstance(dicts[0], float):
            return dicts
        total_dict = {}
        for current_dict in dicts:
            append_dict(total_dict, current_dict, allow_new_keys=True)

        for key, value in total_dict.items():
            value_as_array = np.array(value)
            total_dict[key] = f"{np.mean(value_as_array)} +- {np.std(value_as_array)}"
        return total_dict

    res = {}
    for dataloader_name, ensemble_evals in eval_data.items():

        res[dataloader_name] = aggregate_dicts(list(ensemble_evals.values()))
    return res


def check_for_duplicates(models_config):
    found_model_prefixes = set()
    expected_num_groups = None
    num_groups = len(models_config)
    for key, value in models_config.items():
        if key == "total_groups":
            expected_num_groups = value
            num_groups -= 1
            continue
        model_group_prefix = get_with_assert(
            value,
            "model_group_prefix"
        )
        if model_group_prefix in found_model_prefixes:
            raise Exception(
                f"Duplicates in model prefixes: {model_group_prefix}"
            )
        else:
            found_model_prefixes.add(model_group_prefix)

    if expected_num_groups is not None:
        assert num_groups == expected_num_groups, \
            "Number of model groups is not equal to expected, " \
            "possibly due to duplicates"


# TODO(Alex | 28.03.2024): Move all eval args inside evaluation_kwargs
def make_cross_dict(
    models_dict,
    dataloaders_dict,
    save_path,
    device=torch.device("cuda:0"),
    feature_extractor=None,
    wrap=None,
    verbose=True,
    aggregate=True,
    metrics_mappings=get_metrics_mapping(),
    prune_metrics=["dis"],
    evaluation_func=None,
    evaluation_kwargs={},
    save_after_each_model=True,
    logger=None,
    model_to_prop_dict=None,
    results_key=None,
    recompute_all=False
):

    if not recompute_all and os.path.exists(save_path):
        log_or_print(
            f"Loading existing results from {save_path}",
            logger,
            auto_newline=True
        )
        res = extract_from_pkl(save_path, results_key, assert_non_empty=False)
        # res = torch.load(save_path)??
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        res = {}
    for model_name, models in models_dict.items():

        evaluation_kwargs["model_id"] = model_name

        log_or_print(
            f"evaluating on model: {model_name}",
            logger,
            auto_newline=True
        )
        if model_name in res:
            previous_results = res[model_name]
        else:
            previous_results = {}

        res[model_name] = previous_results

        if isinstance(models, tuple):
            if models[1] == "recompute":
                previous_results = {}
            else:
                assert models[1] == "reuse"
            models = models[0]

        # if model paths are given instead of models
        if isinstance(models[0], str):
            models = make_ensembles(models)

        eval_res = evaluate_ensembles_on_dataloaders(
            models,
            dataloaders_dict,
            device=device,
            feature_extractor=feature_extractor,
            wrap=wrap,
            verbose=False,
            metrics_mappings=metrics_mappings,
            prune_metrics=prune_metrics,
            previous_results=previous_results,
            evaluation_func=evaluation_func,
            evaluation_kwargs=evaluation_kwargs,
            logger=logger
        )

        if aggregate:
            eval_res = aggregate_by_ensemble(eval_res)

        res[model_name] |= eval_res
        res[model_name] |= model_to_prop_dict[model_name]
        if save_after_each_model:
            # torch.save(res, save_path)??
            save_to_pkl(res, save_path, results_key)

    if verbose:
        log_or_print(
            pretty_json(res),
            logger,
            auto_newline=True
        )
    # torch.save(res, save_path)??
    save_to_pkl(res, save_path, results_key)
    return res


def extract_from_pkl(path, key=None, assert_non_empty=True):
    assert os.path.exists(path)
    res = torch.load(path)
    if key is not None:
        # res = res[key]
        if assert_non_empty:
            assert key in res, f"{key} not found in {res}"
        res = res.get(key, {})
    if assert_non_empty:
        assert len(res) > 0
    return res


def save_to_pkl(data, path, key=None):
    pkl_storage = torch.load(path) if os.path.exists(path) else {}
    if key is not None:
        # data = {key: data}
        pkl_storage[key] = data
    else:
        pkl_storage = data
    torch.save(pkl_storage, path)


def evaluate_ensemble(
    ensemble,
    dataloader,
    device=torch.device("cuda:0"),
    feature_extractor=None,
    metrics_mappings=None,
    return_detailed=False,
    prune_metrics=["dis"],
    average_after_softmax=False,
    evaluation_func=None,
    evaluation_kwargs={},
    logger=None
):

    prev_feature_extractor = ensemble.feature_extractor

    if feature_extractor is not None:
        ensemble.feature_extractor = feature_extractor
        ensemble.feature_extractor.to(device)

    ensemble.to(device)
    num_submodels = len(ensemble.submodels)

    prev_weights = ensemble.weights
    ensemble.set_weights([1.0 for _ in range(num_submodels)], normalize=False)
    if average_after_softmax:
        prev_softmax_ensemble = ensemble.softmax_ensemble
        ensemble.softmax_ensemble = True

    if evaluation_func is None:
        res = evaluate_model(
            ensemble,
            dataloader,
            device,
            metrics_mappings=metrics_mappings,
            return_detailed=return_detailed,
            prune_metrics=prune_metrics,
            **evaluation_kwargs
        )
    else:
        res = evaluation_func(
            ensemble,
            dataloader,
            device,
            **evaluation_kwargs
        )
    if average_after_softmax:
        ensemble.softmax_ensemble = prev_softmax_ensemble
    ensemble.set_weights(prev_weights)
    if feature_extractor is not None:
        ensemble.feature_extractor.to(torch.device("cpu"))
    ensemble.feature_extractor = prev_feature_extractor
    ensemble.to("cpu")

    return res


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


# # TODO(Alex | 25.07.2024): re-balance if conditions
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


# TODO(Alex | 15.05.2024): Make it readable
def evaluate_model(
    model,
    dataloader,
    device=torch.device("cuda:0"),
    select_output=None,
    metrics_mappings=None,
    feature_extractor=None,
    return_detailed=False,
    prune_metrics=["dis"],
    metadata_to_group=None,
    logger=None,
    **evaluation_kwargs
):

    def update_correct(
        res,
        key,
        outputs,
        labels,
        detailed_results,
        metadata,
        per_group_totals
    ):

        def add_to_dict(dict, key, subkey, value):
            if key not in dict:
                dict[key] = {}
            if subkey not in dict[key]:
                dict[key][subkey] = []
            dict[key][subkey].append(value)

        def zero_key(d, k):
            if k not in d:
                d[k] = 0

        def pad(arr, bigger_arr, value=0):
            while len(arr) < len(bigger_arr):
                if isinstance(arr, np.ndarray):
                    arr = np.append(arr, [value])
                else:
                    assert isinstance(arr, list)
                    arr.append(value)
            return arr

        zero_key(res, key)

        # masking out samples from ImageNet-A/R
        # that did not have argmax within selected 200 classes
        mask = (outputs.sum(-1) == 0)
        predicted = torch.argmax(outputs, dim=-1)
        predicted[mask] = -1

        if detailed_results is not None:
            probs = get_probs(outputs)
            for i, pred in enumerate(predicted):
                pred = pred.item()
                add_to_dict(detailed_results, key, "pred", pred)
                add_to_dict(detailed_results, key, "conf", probs[i, pred].item())

        if len(predicted.shape) == 1 and len(labels.shape) == 2:
            # ImageNet-hard case, based on this: https://github.com/kirill-vish/Beyond-INet/blob/fd9b1b6c36ecf702fbcc355e037d8e9d307b0137/inference/robustness.py#L117C9-L117C71
            assert predicted.shape[0] == labels.shape[0]
            res[key] += (predicted[:, None] == labels).any(1).sum().item()
        else:
            assert predicted.shape == labels.shape
            correct = (predicted == labels)

            # compute stats for different groups
            if metadata is not None:

                assert metadata_to_group is not None
                groups = metadata_to_group(metadata)

                max_group = groups.max().item()
                num_groups = max_group + 1

                group_acc = np.array([0.0] * num_groups)
                group_count = np.array([0.0] * num_groups)

                denom = []
                processed_data_counts = []

                for current_group in range(num_groups):
                    mask = (groups == current_group).to(correct.device)

                    group_key = key + f'_group_{current_group}'

                    zero_key(res, group_key)
                    zero_key(per_group_totals, group_key)

                    current_group_count = mask.sum().item()
                    group_count[current_group] \
                        = current_group_count
                    num_correct_for_group = (mask * correct).sum().item()
                    group_acc[current_group] \
                        = num_correct_for_group / (current_group_count + int(current_group_count == 0))

                    processed_data_counts.append(per_group_totals[group_key])
                    per_group_totals[group_key] += group_count[current_group]  # for unweighted group accuracy
                    denom.append(per_group_totals[group_key])
                    res[group_key] += num_correct_for_group  # for unweighted group accuracy

                group_wise_key = key + ADJUSTED_GROUPWISE_KEY
                if group_wise_key not in res:
                    res[group_wise_key] = np.array([0] * num_groups)
                else:
                    res_groupwise = res[group_wise_key]
                    processed_data_counts = pad(
                        processed_data_counts,
                        res_groupwise,
                        1
                    )
                    denom = pad(denom, res_groupwise)
                    group_acc = pad(group_acc, res_groupwise)
                    group_count = pad(group_count, res_groupwise)

                denom = np.array(denom)
                processed_data_counts = np.array(processed_data_counts)

                denom += (denom == 0).astype(int)
                prev_weight = processed_data_counts / denom
                curr_weight = group_count / denom

                res[group_wise_key] \
                    = (prev_weight * res[group_wise_key] + curr_weight * group_acc)

            res[key] += (correct).sum().item()

    def prune_metrics_mappings(original_metrics_mappings, keys_to_prune):
        metrics_mappings = []
        for metric_tuple in original_metrics_mappings:
            metric_name = metric_tuple[0]
            if metric_name not in keys_to_prune:
                metrics_mappings.append(metric_tuple)
        return tuple(metrics_mappings)

    def aggregate_over_submodels(res, submodel_values, suffix=''):

        if len(submodel_values) > 0:

            res["best_single_model" + suffix] = max(
                submodel_values
            )
            res["mean_single_model" + suffix] = np.array(submodel_values).mean()

    # Ensure the model is in evaluation mode
    model.to(device)
    model.eval()
    if feature_extractor is not None:
        feature_extractor.to(device)
        feature_extractor.eval()

    # correct = 0
    total = 0

    res = {}

    # detailed results = per sample results
    if return_detailed:
        detailed_results = {}
    else:
        detailed_results = None

    mappings_pruned = False
    per_group_totals = {}

    with torch.no_grad():  # No need to track gradients during evaluation
        for batch_idx, data in enumerate(tqdm(dataloader)):

            inputs = data[0]
            labels = data[1]
            if len(data) > 2:
                assert len(data) in [3, 4]
                metadata = data[2]
            else:
                metadata = None

            inputs, labels = inputs.to(device), labels.to(device)

            if feature_extractor is not None:
                inputs = feature_extractor(inputs)
            if hasattr(model, "soup") and model.soup is not None:
                # TODO(Alex | 13.05.2024): put it inside method model.forward_soup
                if hasattr(model, "feature_extractor") and model.feature_extractor is not None:
                    soup_inputs = model.apply_feature_extractor(inputs)
                else:
                    soup_inputs = inputs
                soup_output = model.soup(soup_inputs)
                update_correct(
                    res,
                    "soup",
                    soup_output,
                    labels,
                    detailed_results,
                    metadata,
                    per_group_totals
                )

            outputs = model(inputs)

            if isinstance(outputs, list):

                assert model.weights is not None, \
                    "Expect ensemble ensemble prediction mode"
                outputs = [output[1] for output in outputs]
                for i, output in enumerate(outputs):
                    if i == len(outputs) - 1:
                        key = f"ensemble"
                    else:
                        key = f"submodel_{i}"
                    update_correct(
                        res,
                        key,
                        output,
                        labels,
                        detailed_results,
                        metadata,
                        per_group_totals
                    )

                submodels_outputs = outputs[:-1]

                if metrics_mappings is not None:

                    # to avoid OOM
                    if prune_metrics and len(submodels_outputs) > 2 and not mappings_pruned:
                        metrics_mappings = prune_metrics_mappings(
                            metrics_mappings,
                            prune_metrics
                        )
                        mappings_pruned = True

                    record_diversity(
                        res,
                        submodels_outputs,
                        torch.stack(submodels_outputs, dim=0),
                        metrics_mappings,
                        labels=labels,
                        detailed_results=detailed_results
                    )

            else:
                update_correct(
                    res,
                    "single_model",
                    outputs,
                    labels,
                    detailed_results,
                    metadata,
                    per_group_totals
                )

            total += labels.size(0)

    keys_to_pop = []

    res_extension_dict = {}

    for key in res:

        if ADJUSTED_GROUPWISE_KEY in key:
            for group_id, value in enumerate(res[key]):
                res_extension_dict[key + f"_group_{group_id}"] = value
            keys_to_pop.append(key)
            continue

        if (
                "ensemble" in key
            or
                "submodel_" in key
            or
                "single_model" == key
            or
                "best_single_model" == key
            or
                "soup" in key
        ):
            if "group" in key:
                divide_by = per_group_totals[key]
            else:
                divide_by = total
        else:
            divide_by = len(dataloader)

        if divide_by == 0:
            assert "group" in key
            keys_to_pop.append(key)
        else:
            res[key] /= divide_by

    for key in keys_to_pop:
        res.pop(key)

    res |= res_extension_dict

    # aggregate to worst groups
    if len(per_group_totals) > 0:
        tmp = {}
        for key in res:

            if "group" in key:
                original_key = key.split("_group")[0]
                if original_key not in tmp:
                    tmp[original_key] = []
                tmp[original_key].append(res[key])

        for key, value in tmp.items():
            res[key + "_worst_group"] = min(value)

    # aggregate to best and mean model
    if len(res) == 1:
        res = res["single_model"]
    else:

        aggregate_over_submodels(
            res,
            [
                value for key, value in res.items()
                    if "submodel" in key and not "group" in key
            ]
        )
        aggregate_over_submodels(
            res,
            [
                value for key, value in res.items()
                    if "submodel" in key and "worst_group" in key
            ],
            suffix="_worst_group"
        )

    if return_detailed:
        return res, detailed_results

    return res


def make_metrics_mappings(eval_type, ood_metric_type):

    if eval_type == "ood_gen":

        metrics_mappings = [
            ("div_different_preds", div_different_preds), # TODO(Alex 06.05.2024): specify by config
            ("div_continous_unique", div_continous_unique),
            ("var", div_var),
        ]

        metrics_mappings += EXTENDED_METRICS
        # if ood_metric_type == "rebuttal_extension":
        #     metrics_mappings += REBUTTAL_METRICS
        # elif ood_metric_type == "a2d_score":
        #     metrics_mappings += [
        #         ("a2d_score", a2d_score)
        #     ]

    else:
        assert eval_type == "ood_det"
        metrics_mappings = [
            ("div_different_preds_per_sample", div_different_preds_per_sample),
            ("div_continous_unique_per_sample", div_continous_unique_per_sample)
        ]
        # if ood_metric_type == "rebuttal_extension":
        #     metrics_mappings += REBUTTAL_METRICS_PER_SAMPLE
        # elif ood_metric_type == "a2d_score":
        #     metrics_mappings += [
        #         ("a2d_score_per_sample", a2d_score_per_sample)
        #     ]
        metrics_mappings += EXTENDED_METRICS_PER_SAMPLE

    return metrics_mappings
