import os
import sys
import torch
import shutil
import sklearn
from stuned.utility.utils import (
    raise_unknown,
    # check_dict,
    get_with_assert,
    # update_dict_by_nested_key,
    pretty_json
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
from diverse_universe.train_eval.losses import (
    # METRICS,
    # div_ortega,
    div_different_preds,
    div_different_preds_per_sample,
    div_continous_unique_per_sample,
    div_continous_unique,
    # div_mean_logit,
    # div_max_logit,
    # div_mean_prob,
    # div_max_prob,
    # div_entropy,
    # div_different_preds_per_model,
    # focal_modifier,
    div_var
)
from diverse_universe.utility.for_notebooks import (
    make_cross_dict,
    # make_ensembles_from_paths,
    # plot_table_from_cross_dict,
    # get_long_table,
    # format_number,
    # make_models_dict_from_huge_string,
    # evaluate_ensembles,
    # ensemble_from_multiple_paths,
    evaluate_ensemble
)
# from diverse_universe.local_models.ensemble import (
#     make_ensembles_from_paths
# )
from diverse_universe.local_models.common import (
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
from diverse_universe.local_datasets.imagenet_c import (
    IN_C_DATALOADERS_NAMES,
    extract_in_c_paths
)
sys.path.pop(0)


# TODO(Alex | 11.05.2024): move to utils
SCRATCH_LOCAL = os.path.join(
    os.sep,
    "scratch_local",
    f"{os.environ.get('USER')}-{os.environ.get('SLURM_JOB_ID')}",
)


IN_VAL_CACHED_2LAYER_PATH = \
    "/mnt/qb/work/oh/arubinstein17/cache/ImageNet1k/val_cache/ed4f10f4ddeb07f1d876_torch_load_block_-1_model_imagenet1k_dataset_1_epochs_50000_samples.hdf5"


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
        eval_type
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
            metrics_mappings=[
                ("div_different_preds", div_different_preds), # TODO(Alex 06.05.2024): specify by config
                ("div_continous_unique", div_continous_unique),
                ("var", div_var)
            ],
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
                "metrics_mappings": [
                    ("div_different_preds_per_sample", div_different_preds_per_sample),
                    ("div_continous_unique_per_sample", div_continous_unique_per_sample)
                ],
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
    models_dict = {}
    model_to_prop_dict = {}
    for model_group_config in models_config.values():
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


def prepare_dataloaders(exp_type, eval_type):
    if exp_type == "deit3b_dataloaders":
        return prepare_deit3b_dataloaders(eval_type)
    elif exp_type == "waterbirds_dataloaders":
        return prepare_waterbirds_dataloaders(eval_type)
    elif exp_type == "debug":
        return prepare_debug_dataloaders()
    else:
        raise_unknown(exp_type, "experiment_type", "prepare_dataloaders")


def prepare_debug_dataloaders():
    # TODO(Alex | 23.07.2024): Move paths to config
    cached_dataloaders_deit3b_dict = make_cached_dataloaders(
        {
            "in_val": IN_VAL_CACHED_2LAYER_PATH,
            "imagenet_a": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/imagenet_a/f37703ce0aea0e55ab2c_torch_load_block_-1_model_imagenet_a_dataset_1_epochs_7500_samples.hdf5",
            "imagenet_r": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/imagenet_r/c9fe46d6544688f46907_torch_load_block_-1_model_imagenet_r_dataset_1_epochs_30000_samples.hdf5"
        }
    )
    deit3b_2layers_all_dataloaders_dict = {
        "imagenet_a": (cached_dataloaders_deit3b_dict["imagenet_a"], "ina"),
        "imagenet_r": (cached_dataloaders_deit3b_dict["imagenet_r"], "inr"),
        "in_val": cached_dataloaders_deit3b_dict["in_val"]
    }
    return deit3b_2layers_all_dataloaders_dict


def subset_dict_by_keys(input_dict, keys):
    return {
        key: input_dict[key] for key in keys
    }


def prepare_waterbirds_dataloaders(eval_type, batch_size=128, num_workers=4):

    waterbirds_config_divdis = {
        "type": "waterbirds",
        'waterbirds': {
            "root_dir": None,
            "keep_metadata": (eval_type == "ood_gen"),
            "use_waterbirds_divdis": True
        }
    }

    waterbirds_train_dataloader_divdis, waterbirds_val_dataloaders_divdis, _ = make_dataloaders(
        waterbirds_config_divdis,
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        to_train=True
    )
    waterbirds_val_dataloaders_divdis["train"] = waterbirds_train_dataloader_divdis

    if eval_type == "ood_gen":

        waterbirds_config = {
            "type": "waterbirds",
            'waterbirds': {
                # "root_dir": "/mnt/qb/work/oh/arubinstein17/cache/wilds/waterbirds",
                "root_dir": SCRATCH_LOCAL,
                # "root_dir": "./",
                "keep_metadata": True,
                # "eval_transforms": "ImageNetEval"
            }
        }

        waterbirds_train_dataloader, waterbirds_val_dataloaders, _ = make_dataloaders(
            waterbirds_config,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
            to_train=True
        )
        waterbirds_val_dataloaders["train"] = waterbirds_train_dataloader

        waterbirds_config_with_transforms = {
            "type": "waterbirds",
            'waterbirds': {
                # "root_dir": "/mnt/qb/work/oh/arubinstein17/cache/wilds/waterbirds",
                "root_dir": SCRATCH_LOCAL,
                # "root_dir": "./",
                "keep_metadata": True,
                "eval_transforms": "ImageNetEval"
            }
        }

        waterbirds_train_dataloader_with_transforms, waterbirds_val_dataloaders_with_transforms, _ = make_dataloaders(
            waterbirds_config_with_transforms,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
            to_train=True
        )

        waterbirds_val_dataloaders_with_transforms["train"] = waterbirds_train_dataloader_with_transforms

        all_waterbirds_dataloaders_dict = (
                waterbirds_val_dataloaders
            |
                update_key_names(
                    waterbirds_val_dataloaders_with_transforms,
                    "_with_transform"
                )
        )
    else:
        assert eval_type == "ood_det"
        all_waterbirds_dataloaders_dict = {}

    all_waterbirds_dataloaders_dict |= update_key_names(
        waterbirds_val_dataloaders_divdis,
        "_divdis"
    )

    return all_waterbirds_dataloaders_dict


def update_key_names(d, suffix):
    return {
        key + suffix: value
            for key, value
                in d.items()
    }


def prepare_deit3b_dataloaders(eval_type):

    # assert exp_type == "deit3b"
    # Deit3B

    # TODO(Alex | 23.07.2024): Move paths to config
    in_c_dataloaders = extract_in_c_paths(
        "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/",
        [1, 5],
        "torch_load_block_-1_model"
    )

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

    cached_dataloaders_deit3b_dict = make_cached_dataloaders(
        {
            "in_val": IN_VAL_CACHED_2LAYER_PATH,
        } | in_c_dataloaders
    )

    in_c_dataloaders_dict = subset_dict_by_keys(
        cached_dataloaders_deit3b_dict,
        in_c_dataloaders.keys()
    )

    deit3b_2layers_all_dataloaders_dict = {
        "in_val": cached_dataloaders_deit3b_dict["in_val"]
    }

    # TODO(Alex | 23.07.2024): Move paths to config
    cached_dataloaders_deit3b_dict |= make_cached_dataloaders(
        {
            "iNaturalist": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/iNaturalist/214988c198612f2682e6_torch_load_block_-1_model_iNaturalist_dataset_1_epochs_10000_samples.hdf5",
            "OpenImages": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/openimages/816f6e9e7fa686d093e8_torch_load_block_-1_model_openimages_dataset_1_epochs_17632_samples.hdf5"
        }
    )

    if eval_type == "ood_gen":

        # TODO(Alex | 23.07.2024): Move paths to config
        in_d_dataloaders = {
            "in_d_background": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/background/656ee9569bfbcf377ea9_torch_load_block_-1_model_background_dataset_1_epochs_3764_samples.hdf5",
            "in_d_texture": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/texture/8ecfa96f9135692c2cf2_torch_load_block_-1_model_texture_dataset_1_epochs_498_samples.hdf5",
            "in_d_material": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/material/116c69294993da7f1b7f_torch_load_block_-1_model_material_dataset_1_epochs_573_samples.hdf5"
        }

        # TODO(Alex | 23.07.2024): Move paths to config
        cached_dataloaders_deit3b_dict |= make_cached_dataloaders(
            {
            "imagenet_a": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/imagenet_a/f37703ce0aea0e55ab2c_torch_load_block_-1_model_imagenet_a_dataset_1_epochs_7500_samples.hdf5",
            "imagenet_r": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/imagenet_r/c9fe46d6544688f46907_torch_load_block_-1_model_imagenet_r_dataset_1_epochs_30000_samples.hdf5",
            "mixed_rand": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/mixed_rand/bf802cc68d4a8b524c38_torch_load_block_-1_model_mixed_rand_dataset_1_epochs_4050_samples.hdf5",
            "mixed_same": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/mixed_same/03839176c08efd3bf9e0_torch_load_block_-1_model_mixed_same_dataset_1_epochs_4050_samples.hdf5",
            "only_bg_t": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/only_bg_t/82c8996731b022fe6a5f_torch_load_block_-1_model_only_bg_t_dataset_1_epochs_4050_samples.hdf5",
            "no_fg": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/no_fg/5cc8e5036cec4c12a397_torch_load_block_-1_model_no_fg_dataset_1_epochs_4050_samples.hdf5",
            "mixed_next": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/mixed_next/0cbfd71928f9d1f69cd2_torch_load_block_-1_model_mixed_next_dataset_1_epochs_4050_samples.hdf5",
            "only_fg": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/only_fg/3de26782660a000fb937_torch_load_block_-1_model_only_fg_dataset_1_epochs_4050_samples.hdf5",
            "original": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/original/d5967a112b317104ccb5_torch_load_block_-1_model_original_dataset_1_epochs_4050_samples.hdf5",
            "only_bg_b": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/only_bg_b/b2005edd362a19e1ae41_torch_load_block_-1_model_only_bg_b_dataset_1_epochs_4050_samples.hdf5",
            "stylized": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/stylized/6a559f1b1becae879c79_torch_load_block_-1_model_stylized_dataset_1_epochs_800_samples.hdf5",
            "sketch": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/sketch/ff0161b6b924453b0c68_torch_load_block_-1_model_sketch_dataset_1_epochs_800_samples.hdf5",
            "edge": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/edge/7f7ea165dfb604e47f3c_torch_load_block_-1_model_edge_dataset_1_epochs_160_samples.hdf5",
            "silhouette": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/silhouette/9def5ec3aecf90d0515c_torch_load_block_-1_model_silhouette_dataset_1_epochs_160_samples.hdf5",
            "cue-conflict": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/cue-conflict/5f1aa995aa7afccf46e8_torch_load_block_-1_model_cue-conflict_dataset_1_epochs_1280_samples.hdf5",
            # "in_val": in_val_cached_2layer_path,
            } | in_d_dataloaders
        )

        deit3b_2layers_all_dataloaders_dict |= {
            "imagenet_a": (cached_dataloaders_deit3b_dict["imagenet_a"], "ina"),
            "imagenet_r": (cached_dataloaders_deit3b_dict["imagenet_r"], "inr"),
            "mixed_rand": (cached_dataloaders_deit3b_dict["mixed_rand"], "IN9"),
            "mixed_next": (cached_dataloaders_deit3b_dict["mixed_next"], "IN9"),
            "cue-conflict": (cached_dataloaders_deit3b_dict["cue-conflict"], "mvh"),
            "stylized": (cached_dataloaders_deit3b_dict["stylized"], "mvh"),
            "iNaturalist": cached_dataloaders_deit3b_dict["iNaturalist"],
            "OpenImages": cached_dataloaders_deit3b_dict["OpenImages"]
            # "in_val": cached_dataloaders_deit3b_dict["in_val"],
            # "in_val_5k_hard": in_val_5k_hard,
            # "in_val_45k_easy": in_val_45k_easy,
            # "in_train_3068_hard": in_train_3068_hard,
            # "fog_1": cached_dataloaders_deit3b_dict["fog_1"],
            # "fog_5": cached_dataloaders_deit3b_dict["fog_5"],
            # "defocus_blur_1": cached_dataloaders_deit3b_dict["defocus_blur_1"],
            # "defocus_blur_5": cached_dataloaders_deit3b_dict["defocus_blur_5"],
        }

        # deit3b_2layers_all_dataloaders_dict |= {
        #     key: cached_dataloaders_deit3b_dict[key] for key in in_c_dataloaders.keys()
        # }

        # deit3b_2layers_all_dataloaders_dict |= in_c_dataloaders_dict

        deit3b_2layers_all_dataloaders_dict |= subset_dict_by_keys(
            cached_dataloaders_deit3b_dict,
            in_d_dataloaders.keys()
        )

        # deit3b_2layers_all_dataloaders_dict |= {
        #     key: cached_dataloaders_deit3b_dict[key] for key in in_d_dataloaders.keys()
        # }
    else:
        assert eval_type == "ood_det"
        # cached_dataloaders_deit3b_dict |= make_cached_dataloaders(
        #     {
        #         "iNaturalist": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/iNaturalist/214988c198612f2682e6_torch_load_block_-1_model_iNaturalist_dataset_1_epochs_10000_samples.hdf5",
        #         "OpenImages": "/mnt/qb/work/oh/arubinstein17/cache/val_datasets/openimages/816f6e9e7fa686d093e8_torch_load_block_-1_model_openimages_dataset_1_epochs_17632_samples.hdf5"
        #     }
        # )
        deit3b_2layers_all_dataloaders_dict |= {
            # "imagenet_a": (cached_dataloaders_deit3b_dict["imagenet_a"], "ina"),
            # "imagenet_r": (cached_dataloaders_deit3b_dict["imagenet_r"], "inr"),
            # "mixed_rand": (cached_dataloaders_deit3b_dict["mixed_rand"], "IN9"),
            # "mixed_next": (cached_dataloaders_deit3b_dict["mixed_next"], "IN9"),
            # "cue-conflict": (cached_dataloaders_deit3b_dict["cue-conflict"], "mvh"),
            # "in_val": cached_dataloaders_deit3b_dict["in_val"],
            "iNaturalist": cached_dataloaders_deit3b_dict["iNaturalist"],
            "OpenImages": cached_dataloaders_deit3b_dict["OpenImages"]
            # "in_train_3068_hard": in_train_3068_hard
        }
    deit3b_2layers_all_dataloaders_dict |= in_c_dataloaders_dict

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
def compute_roc_auc(detailed_res_id, detailed_res_ood, verbose=True):

    labels_div, divs = joint_res(
        detailed_res_id,
        detailed_res_ood, # class 1 has higher div
        'div_different_preds_per_sample'
    )

    labels_cont_uniq, cont_uniq = joint_res(
        detailed_res_id,
        detailed_res_ood, # class 1 has higher div
        'div_continous_unique_per_sample'
    )

    labels_conf, confs = joint_res(
        detailed_res_ood["ensemble"],
        detailed_res_id["ensemble"], # class 1 has higher conf
        'conf'
    )

    # labels_conf_sm, confs_sm = joint_res(
    #     ce_a2d_detailed_res_sm["ensemble"],
    #     ce_a2d_detailed_res_in_val_sm["ensemble"], # class 1 has higher conf
    #     'conf'
    # )

    labels_conf_0, confs_0 = joint_res(
        detailed_res_ood["submodel_0"],
        detailed_res_id["submodel_0"], # class 1 has higher conf
        'conf'
    )

    roc_auc_conf = sklearn.metrics.roc_auc_score(labels_conf, confs)
    # roc_auc_conf_sm = sklearn.metrics.roc_auc_score(labels_conf_sm, confs_sm)
    roc_auc_conf_0 = sklearn.metrics.roc_auc_score(labels_conf_0, confs_0)
    roc_auc_divs = sklearn.metrics.roc_auc_score(labels_div, divs)
    roc_auc_cont_uniq = sklearn.metrics.roc_auc_score(labels_cont_uniq, cont_uniq)


    ##########
    # TODO(Alex | 28.03.2024): remove duplication above and do it in a loop

    res = {
        "ensemble": roc_auc_conf,
        "submodel_0": roc_auc_conf_0,
        "divs": roc_auc_divs,
        "cont_unique": roc_auc_cont_uniq
    }

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
