# import os
import sys
import copy
from stuned.utility.logger import (
    LOGGING_CONFIG_KEY,
    GDRIVE_FOLDER_KEY,
    # make_logger
)
from stuned.utility.utils import (
    check_dict,
    get_with_assert,
    update_dict_by_nested_key,
    get_project_root_path
)
from stuned.local_datasets.imagenet1k import (
    IMAGENET_KEY
)
from stuned.utility.configs import (
    find_nested_keys_by_keyword_in_config,
    normalize_paths
)


# local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, get_project_root_path())
# from utility.utils import (
#     raise_unknown,
#     check_dict,
#     get_with_assert,
#     update_dict_by_nested_key
# )
# from utility.logger import (
#     LOGGING_CONFIG_KEY,
#     GDRIVE_FOLDER_KEY,
#     make_logger
# )
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
from diverse_universe.local_datasets.common import (
    UNLABELED_DATASET_KEY
)
# from local_datasets.tiny_imagenet import (
#     TINY_IMAGENET,
#     TINY_IMAGENET_TEST
# )
from diverse_universe.local_datasets.from_h5 import (
    FROM_H5
)
from diverse_universe.train.losses import (
    DIVDIS_LOSS_KEY,
    LAMBDA_KEY
)
# from local_algorithms.div_dis import (
#     DIVDIS_DIVERSITY_WEIGHT_KEY
# )
# from local_datasets.imagenet1k import (
#     IMAGENET_KEY
# )
sys.path.pop(0)


OUTPUT_CSV_KEY = "output_csv"
# DEFAULT_WANDB_CONFIG = \
#     {
#         "netrc_path": "~/.netrc",
#         "stats": {
#             "input": True,
#             "logits": True,
#             "prediction": True,
#             "target": True
#         }
#     }
DEFAULT_SETUP = "default"
# WILDS_SETUP = "wilds_algorithm"
# ALGOS_WITH_UNLABELED_DATA = (DIV_DIS_ALGO_NAME,)
DIVERSITY_LAMBDA_KEY = "diversity_lambda"
# DIVERSE_GRADIENTS_LOSS_COMPATIBLE_MODELS = (
#     REDNECK_ENSEMBLE_KEY,
#     "diverse_vit_switch_ensemble"
# )


def check_exp_config(experiment_config, config_path, logger=None):
    pass


# def check_exp_config(experiment_config, config_path, logger=make_logger()):

#     main_config_keys = [
#         "cache_path",
#         "data",
#         "params",
#         "statistics",
#         "model",
#         "experiment_name",
#         "start_time",
#         "current_run_folder"
#     ]
#     optional_config_keys = [
#         "autogenerated_params",
#         "logging",
#         "patch",
#         "wandb_sweep"
#     ]
#     if not experiment_config.get("use_hardcoded_config", False):
#         check_dict(
#             experiment_config,
#             main_config_keys,
#             optional_keys=optional_config_keys
#         )
#         data_config = experiment_config["data"]
#         check_data_config(data_config)
#         if "checkpoint" in experiment_config:
#             check_checkpoint_config(experiment_config["checkpoint"])
#         params_config = experiment_config["params"]
#         check_params_config(params_config)
#         check_statistics_config(experiment_config["statistics"])
#         model_config = experiment_config["model"]
#         check_model_config(model_config)

#         if "autogenerated_params" in experiment_config:
#             check_autogenerated_params(
#                 experiment_config["autogenerated_params"]
#             )
#         if params_config["to_train"]:
#             check_config_consistency(
#                 model_config,
#                 params_config,
#                 data_config
#             )


def patch_exp_config(experiment_config):

    def assert_key(key, config):
        assert key in config, f"{key} is expected to be in {config}"

    if (
        LOGGING_CONFIG_KEY not in experiment_config
            or not complete_logging_config(
                experiment_config[LOGGING_CONFIG_KEY]
            )
    ):

        output_config = None
        gdrive_folder = None
        if LOGGING_CONFIG_KEY in experiment_config:
            logging_config = experiment_config[LOGGING_CONFIG_KEY]
            # assert_key(GDRIVE_FOLDER_KEY, logging_config)
            assert_key(OUTPUT_CSV_KEY, logging_config)
            if logging_config[OUTPUT_CSV_KEY] is not None:
                check_csv_line_config(
                    logging_config[OUTPUT_CSV_KEY]
                )
            output_config \
                = logging_config[OUTPUT_CSV_KEY]
            # gdrive_folder = logging_config[GDRIVE_FOLDER_KEY]
            gdrive_folder = logging_config.get(GDRIVE_FOLDER_KEY)

        logging_config = copy.deepcopy(experiment_config["statistics"])
        logging_config.pop("batchwise")
        logging_config.pop("keep_modelwise")
        if output_config is not None:
            logging_config[OUTPUT_CSV_KEY] = output_config
        logging_config[GDRIVE_FOLDER_KEY] = gdrive_folder
        experiment_config[LOGGING_CONFIG_KEY] = logging_config
    else:
        assert (
                experiment_config["statistics"]["use_wandb"]
            ==
                experiment_config[LOGGING_CONFIG_KEY]["use_wandb"]
        )
        assert (
                experiment_config["statistics"]["use_tb"]
            ==
                experiment_config[LOGGING_CONFIG_KEY]["use_tb"]
        )

    patches_config = experiment_config.get("patch", {})
    diversity_lambda = patches_config.get(DIVERSITY_LAMBDA_KEY)
    if diversity_lambda is not None:
        params_config = get_with_assert(experiment_config, "params")
        patch_diversity_lambda(params_config, diversity_lambda)

    unlabeled_data = patches_config.get(UNLABELED_DATASET_KEY)
    if unlabeled_data is not None:
        data_config = get_with_assert(experiment_config, "data")
        patch_unlabeled_data(data_config, unlabeled_data)

    paths_in_config = []
    for path_keyword in ["root"]:
        paths_in_config += find_nested_keys_by_keyword_in_config(
            experiment_config,
            path_keyword
        )
    normalize_paths(experiment_config, paths_in_config)


def patch_unlabeled_data(data_config, unlabeled_data):

    def get_indices_to_keep(dataset_config):
        dataset_type = get_with_assert(dataset_config, "type")
        if dataset_type in [IMAGENET_KEY, FROM_H5]:
            specific_config = get_with_assert(dataset_config, dataset_type)
            indices_to_keep = specific_config.get("indices_to_keep")
        else:
            indices_to_keep = None
        return indices_to_keep

    dataset_config = get_with_assert(data_config, "dataset")
    if "unlabeled_dataset" in data_config:
        # TODO(Alex | 13.02.2024): remove duplication and merge with else
        unlabeled_dataset_config = get_with_assert(data_config, "unlabeled_dataset")
        dataset_config = get_with_assert(data_config, "dataset")
        indices_to_keep_main = get_indices_to_keep(dataset_config)
        indices_to_keep_unlabeled = get_indices_to_keep(unlabeled_dataset_config)
        if indices_to_keep_main is not None:
            assert indices_to_keep_unlabeled is not None
            if unlabeled_data.get("same_indices", True):
                assert indices_to_keep_main == indices_to_keep_unlabeled

    else:
        unlabeled_dataset_config = copy.deepcopy(dataset_config)
        unlabeled_dataset_config["split"] = get_with_assert(unlabeled_data, "split")
        dataset_type = get_with_assert(unlabeled_dataset_config, "type")
        if dataset_type in [IMAGENET_KEY, FROM_H5]:
            unlabeled_dataset_config_for_type = unlabeled_dataset_config[dataset_type]
            assert unlabeled_dataset_config_for_type.get("reverse_indices", {}).get("train", True), \
                "For main dataset indices should be reversed or not given"
            unlabeled_dataset_config_for_type["reverse_indices"] = unlabeled_data.get(
                "reverse_indices",
                {"train": False}
            )
            # so that unlabeled dataset is not unloading and loading every time
            if dataset_type == FROM_H5:
                unlabeled_dataset_config_for_type["total_samples"] \
                    = {"train": get_with_assert(unlabeled_dataset_config_for_type, "max_chunk")}
                assert len(unlabeled_dataset_config_for_type.get("indices_to_keep", {})) > 0, \
                    "Should have indices_to_keep for unlabeled data"
    data_config[UNLABELED_DATASET_KEY] = unlabeled_dataset_config


def patch_diversity_lambda(params_config, diversity_lambda):

    def patch_weights(
        task_loss_weight_nested_key,
        weights_nested_key,
        diversity_lambda,
        as_list=True
    ):

        def make_error_msg(arg_name, equal_to):
            return (
                f"Should have {arg_name} == {equal_to}, "
                f"when using {DIVERSITY_LAMBDA_KEY}"
            )

        gradient_based_losses_weights = get_with_assert(
            params_config,
            weights_nested_key
        )
        none_value = [None] if as_list else None
        error_msg = make_error_msg(weights_nested_key[-1], str(none_value))
        if as_list:
            assert gradient_based_losses_weights == none_value, error_msg
        else:
            assert gradient_based_losses_weights is none_value, error_msg

        task_loss_weight = get_with_assert(
            params_config,
            task_loss_weight_nested_key
        )
        assert task_loss_weight is None, \
            make_error_msg(task_loss_weight_nested_key[-1], "None")

        if as_list:
            if diversity_lambda == 0:
                new_weights = []
            else:
                new_weights = [diversity_lambda]
        else:
            new_weights = diversity_lambda

        update_dict_by_nested_key(
            params_config,
            weights_nested_key,
            new_weights
        )

        update_dict_by_nested_key(
            params_config,
            task_loss_weight_nested_key,
            (1 - diversity_lambda)
        )

    assert diversity_lambda >= 0 and diversity_lambda <= 1, \
        f"{DIVERSITY_LAMBDA_KEY} should be in [0, 1]"

    setup = params_config.get("setup", DEFAULT_SETUP)
    # if setup == DEFAULT_SETUP:
    assert setup == DEFAULT_SETUP
    criterion_nested_key = ["train", "criterion"]
    criterion_config = get_with_assert(
        params_config,
        criterion_nested_key
    )
    # if DIVERSE_GRADIENTS_LOSS_KEY in criterion_config:
    #     diverse_loss_nested_key = criterion_nested_key + [
    #         DIVERSE_GRADIENTS_LOSS_KEY
    #     ]
    #     task_loss_weight_nested_key \
    #         = diverse_loss_nested_key + [TASK_LOSS_WEIGHT_KEY]
    #     weights_nested_key \
    #         = diverse_loss_nested_key + [GRADIENT_BASED_LOSSES_WEIGHTS_KEY]

    #     patch_weights(
    #         task_loss_weight_nested_key,
    #         weights_nested_key,
    #         diversity_lambda,
    #         as_list=True
    #     )

    #     if diversity_lambda == 0:

    #         gradient_based_losses_nested_key \
    #             = diverse_loss_nested_key + [GRADIENT_BASED_LOSSES_KEY]

    #         update_dict_by_nested_key(
    #             params_config,
    #             gradient_based_losses_nested_key,
    #             []
    #         )
    # else:

    divdis_config = get_with_assert(
        criterion_config,
        DIVDIS_LOSS_KEY
    )

    assert divdis_config.get(LAMBDA_KEY) is None, \
        f"Should have {LAMBDA_KEY} == None, when patching diversity lambda"

    divdis_config[LAMBDA_KEY] = diversity_lambda
    # else:
    #     assert setup == WILDS_SETUP
    #     divdis_config_nested_key = [
    #         "algorithm",
    #         "div_dis_algorithm"
    #     ]

    #     classification_loss_weight_nested_key \
    #         = divdis_config_nested_key + [CLASSIFICATION_LOSS_WEIGHT_KEY]
    #     divdis_weight_nested_key \
    #         = divdis_config_nested_key + [DIVDIS_DIVERSITY_WEIGHT_KEY]

    #     patch_weights(
    #         classification_loss_weight_nested_key,
    #         divdis_weight_nested_key,
    #         diversity_lambda,
    #         as_list=False
    #     )


def complete_logging_config(logging_config):
    return check_logging_config(logging_config, raise_if_wrong=False)


def check_logging_config(logging_config, raise_if_wrong=True):

    check_1 = False
    check_2 = False

    check_1 = check_dict(logging_config,
        [
            OUTPUT_CSV_KEY,
            GDRIVE_FOLDER_KEY,
            "use_wandb",
            "use_tb"
        ],
        optional_keys=[
            "wandb",
            "tb",
        ],
        check_reverse=True,
        raise_if_wrong=raise_if_wrong
    )

    if check_1:
        if logging_config["use_wandb"]:

            check_2 = "wandb" in logging_config

            if raise_if_wrong:
                assert check_2, f"{logging_config} is expected to conain \"wandb\""

            if check_2:
                check_2 = check_dict(
                    logging_config["wandb"],
                    [
                        "netrc_path"
                    ],
                    check_reverse=True,
                    raise_if_wrong=raise_if_wrong
                )
        else:
            check_2 = True

        # if check_2 and logging_config["use_tb"]:

        #     check_2 = "tb" in logging_config

        #     if raise_if_wrong:
        #         assert check_2, f"{logging_config} is expected to conain \"tb\""

        #     if check_2:
        #         check_2 = check_tb_config(logging_config["tb"])

    return check_1 and check_2


# def check_config_consistency(model_config, params_config, data_config):

#     def expect_false(
#         config,
#         expected_false_name,
#         incompatible_name
#     ):
#         assert not config.get(
#             expected_false_name,
#             False
#         ), \
#             f"Cannot use {expected_false_name} with " \
#             f"{incompatible_name} in config: \n{config}"

#     def mutually_exclusive(
#         config,
#         first_option,
#         second_option
#     ):
#         if config.get(
#             first_option,
#             False
#         ):
#             expect_false(
#                 config,
#                 second_option,
#                 first_option
#             )
#         if config.get(
#             second_option,
#             False
#         ):
#             expect_false(
#                 config,
#                 first_option,
#                 second_option
#             )

#     def pairwise_mutually_exclusive(config, options):
#         assert isinstance(options, list)
#         for i in range(len(options)):
#             for j in range(i + 1, len(options)):
#                 mutually_exclusive(
#                     config,
#                     options[i],
#                     options[j]
#                 )

#     setup = params_config.get("setup", DEFAULT_SETUP)
#     model_type = model_config["type"]
#     model_specific_config = get_with_assert(
#         model_config,
#         model_type
#     )
#     pairwise_mutually_exclusive(
#         model_specific_config,
#         [SINGLE_MODEL_KEY, POE_KEY, "weights"]
#     )
#     criterion_config = params_config["train"]["criterion"]
#     if setup == DEFAULT_SETUP:

#         criterion_type = get_with_assert(criterion_config, "type")
#         if criterion_type == DIVERSE_GRADIENTS_LOSS_KEY:
#             assert model_type in DIVERSE_GRADIENTS_LOSS_COMPATIBLE_MODELS, \
#                 f"Model type should be in " \
#                 f"{DIVERSE_GRADIENTS_LOSS_COMPATIBLE_MODELS} " \
#                 f"when using {DIVERSE_GRADIENTS_LOSS_KEY}"

#             expect_false(
#                 model_specific_config,
#                 SINGLE_MODEL_KEY,
#                 DIVERSE_GRADIENTS_LOSS_KEY
#             )
#             expect_false(
#                 model_specific_config,
#                 POE_KEY,
#                 DIVERSE_GRADIENTS_LOSS_KEY
#             )

#         if model_type == REDNECK_ENSEMBLE_KEY:
#             if (
#                     not model_specific_config.get(
#                         SINGLE_MODEL_KEY,
#                         False
#                     )
#                 and
#                     not model_specific_config.get(
#                         POE_KEY,
#                         False
#                     )
#             ):
#                 allowed_types = [DIVERSE_GRADIENTS_LOSS_KEY, DIVDIS_LOSS_KEY]
#                 if model_config[REDNECK_ENSEMBLE_KEY]["n_models"] == 1:
#                     allowed_types.append("xce")
#                 assert criterion_type in allowed_types, \
#                     f"Criterion type should be in {allowed_types} " \
#                     f"when using {REDNECK_ENSEMBLE_KEY} " \
#                     f"without \"{SINGLE_MODEL_KEY}\" option" \
#                     f"or \"{POE_KEY}\" option"

#     elif setup == WILDS_SETUP:

#         algorithm_config = get_with_assert(params_config, "algorithm")
#         algorithm_type = get_with_assert(algorithm_config, "type")
#         if algorithm_type in ALGOS_WITH_UNLABELED_DATA:
#             assert UNLABELED_DATASET_KEY in data_config, \
#                 f"Need unlabeled data for algorithm from " \
#                 f"{ALGOS_WITH_UNLABELED_DATA}"
#     else:
#         raise_unknown("setup", setup, "params_config")


# def check_data_config(data_config):
#     check_dict(data_config,
#         [
#             "num_data_readers",
#             "dataset"
#         ],
#         optional_keys=[UNLABELED_DATASET_KEY],
#         check_reverse=True
#     )
#     check_dataset_config(data_config["dataset"])


# def check_dataset_config(dataset_config):
#     check_dict(dataset_config,
#         [
#             "type"
#         ],
#         check_reverse=False
#     )
#     dataset_type = dataset_config["type"]
#     if dataset_type == "dsprites":
#         check_config_for_type(
#             dataset_config,
#             dataset_type,
#             check_dsprites_data_config
#         )
#     elif dataset_type == TINY_IMAGENET:
#         check_config_for_type(
#             dataset_config,
#             dataset_type,
#             check_tiny_imagenet_data_config
#         )
#     elif dataset_type == "features_labeller":
#         check_config_for_type(
#             dataset_config,
#             dataset_type,
#             check_features_labeller_data_config
#         )
#     elif dataset_type == "synthetic_pos":
#         check_config_for_type(
#             dataset_config,
#             dataset_type,
#             check_synthetic_pos_data_config
#         )
#     elif dataset_type in [
#         "waterbirds",
#         "camelyon17"
#     ]:
#         pass
#     else:
#         assert dataset_type in dataset_config, \
#             f"No {dataset_type} in \n{dataset_config}\n"


# def check_config_for_type(config, type_name, check_func, raise_if_wrong=True):

#     check_1 = False
#     check_2 = False

#     check_1 = check_dict(config,
#         [
#             type_name
#         ],
#         raise_if_wrong=raise_if_wrong
#     )
#     if check_1:
#         check_2 = check_func(config[type_name])
#     return check_1 and check_2


# def check_features_labeller_data_config(features_labeller_data_config):
#     check_dict(features_labeller_data_config,
#         [
#             "features_list",
#             "num_classes_per_feature",
#             "num_samples_per_cell",
#             "base_data"
#         ],
#         optional_keys=[
#             "off_diag_percent",
#             "single_label",
#             "manual_values_per_class"
#         ],
#         check_reverse=True
#     )
#     check_dataset_config(features_labeller_data_config["base_data"])


# def check_synthetic_pos_data_config(synthetic_pos_data_config):
#     check_dict(synthetic_pos_data_config,
#         [
#             "dim_for_label",
#             "num_train_images",
#             "num_test_images",
#             "img_size",
#             "fill_object_with_ones"
#         ],
#         check_reverse=True
#     )


# def check_dsprites_data_config(dsprites_config):
#     check_dict(dsprites_config,
#         [
#             "type",
#             "train_val_split",
#             "path"
#         ],
#         optional_keys=[
#             "transforms",
#             "color_scheme",
#             "file_with_pruned_indices_path"
#         ],
#         check_reverse=True
#     )


# def check_tiny_imagenet_data_config(tiny_imagenet_config):
#     check_dict(tiny_imagenet_config,
#         [
#             "total_number_of_samples",
#             "train_val_split",
#             "normalize",
#             "path"
#         ],
#         check_reverse=True
#     )
#     assert_type(tiny_imagenet_config["normalize"], bool)


# def check_checkpoint_config(checkpoint_config):
#     check_dict(checkpoint_config,
#         [
#             "starting_checkpoint_path"
#         ],
#         optional_keys=["load_only_model", "check_only_model"],
#         check_reverse=True
#     )


# def check_params_config(params_config):
#     check_dict(params_config,
#         [
#             "to_profile",
#             "to_train",
#             "to_eval",
#             "random_seed",
#             "use_gpu",
#             "metric"
#         ],
#         optional_keys=["train", "eval", "setup", "algorithm"],
#         check_reverse=True
#     )
#     assert_type(params_config["to_profile"], bool)
#     assert_type(params_config["use_gpu"], bool)
#     check_metric_config(params_config["metric"])
#     to_train = params_config["to_train"]
#     to_eval = params_config["to_eval"]
#     assert_type(to_train, bool)
#     assert_type(to_eval, bool)
#     if not (to_train or to_eval):
#         raise Exception(
#             "At least one of these options should"
#             " be True: \"to_train\", \"to_eval\""
#         )
#     if to_train:
#         check_config_for_type(
#             params_config,
#             "train",
#             check_train_config
#         )

#     if to_eval:
#         check_config_for_type(
#             params_config,
#             "eval",
#             check_eval_config
#         )


# def check_train_config(train_config):
#     check_dict(train_config,
#         [
#             "freeze_model_on_first_epoch",
#             "n_epochs",
#             "reset_epochs",
#             "batch_size",
#             "optimizer",
#             "criterion"
#         ],
#         optional_keys=[
#             "off_diag_supervision",
#             "checkpoints_to_save",
#             "unlabeled_criterion"
#         ],
#         check_reverse=True
#     )
#     assert_type(train_config["freeze_model_on_first_epoch"], bool)
#     check_optimizer_config(train_config["optimizer"])
#     check_criterion_config(train_config["criterion"])


# def check_eval_config(eval_config):
#     check_dict(eval_config,
#         [
#             "batch_size",
#             "eval_only_last_epoch"
#         ],
#         check_reverse=True
#     )
#     assert_type(eval_config["eval_only_last_epoch"], bool)


# def check_metric_config(metric_config):
#     check_dict(
#         metric_config,
#         ["type"],
#         optional_keys=[
#             "accuracy_top_k",
#             "final_aggregation",
#             "aggregatable_stages",
#             "report_best_metric"
#         ],
#         check_reverse=True
#     )


# def check_optimizer_config(optimizer_config):
#     check_dict(
#         optimizer_config,
#         [
#             "type",
#             "start_lr"
#         ],
#         optional_keys=[
#             "lr_scheduler",
#             "sgd",
#             "adam",
#             "nan_checking",
#             "adamW"
#         ],
#         check_reverse=True
#     )
#     if "lr_scheduler" in optimizer_config:
#         check_lr_scheduler_config(optimizer_config["lr_scheduler"])
#     optimizer_type = optimizer_config["type"]
#     if optimizer_type == "sgd":
#         check_config_for_type(
#             optimizer_config,
#             optimizer_type,
#             check_sgd_config
#         )
#     elif optimizer_type == "adam":
#         pass
#     elif optimizer_type == "adamW":
#         pass
#     else:
#         raise_unknown(
#             "optimizer type",
#             optimizer_type,
#             "optimizer config"
#         )


# def check_lr_scheduler_config(lr_scheduler_config):
#     check_dict(
#         lr_scheduler_config,
#         [
#             "type"
#         ]
#     )
#     lr_scheduler_type = lr_scheduler_config["type"]
#     if lr_scheduler_type == "reduce_on_plateau":
#         pass
#     elif lr_scheduler_type == "constant":
#         check_config_for_type(
#             lr_scheduler_config,
#             lr_scheduler_type,
#             check_constant_lr_config
#         )
#     elif lr_scheduler_type.startswith(FROM_CLASS_KEY):
#         pass
#     else:
#         raise_unknown(
#             "lr_scheduler type",
#             lr_scheduler_type,
#             "lr_scheduler config"
#         )


# def check_reduce_on_plateau_config(reduce_on_plateau_config):
#     check_dict(
#         reduce_on_plateau_config,
#         [
#             "factor",
#             "patience"
#         ],
#         check_reverse=True
#     )


# def check_constant_lr_config(constant_lr_config):
#     check_dict(
#         constant_lr_config,
#         [
#             "factor",
#             "total_iters"
#         ],
#         check_reverse=True
#     )


# def check_sgd_config(sgd_config):
#     check_dict(
#         sgd_config,
#         [
#             "momentum",
#             "weight_decay"
#         ],
#         check_reverse=True
#     )


# def check_criterion_config(criterion_config):
#     check_dict(
#         criterion_config,
#         ["type"],
#         check_reverse=True,
#         optional_keys=[
#             DIVERSE_GRADIENTS_LOSS_KEY,
#             DIVDIS_LOSS_KEY,
#             "smoothing_eps"
#         ]
#     )
#     if criterion_config["type"] \
#         == DIVERSE_GRADIENTS_LOSS_KEY:

#         check_dict(
#             criterion_config,
#             [
#                 DIVERSE_GRADIENTS_LOSS_KEY
#             ]
#         )
#         check_diverse_gradients_loss_criterion_config(
#             criterion_config[DIVERSE_GRADIENTS_LOSS_KEY]
#         )


# def check_diverse_gradients_loss_criterion_config(criterion_subconfig):
#     check_dict(
#         criterion_subconfig,
#         [
#             "task_loss",
#             "gradient_based_losses",
#             "gradient_based_losses_weights"
#         ],
#         check_reverse=True,
#         optional_keys=[
#             ORTHOGONAL_GRADIENTS_LOSS,
#             ON_MANIFOLD_LOSS,
#             DIV_VIT_LOSS,
#             "task_loss_weight",
#             "use_max_pred"
#         ]
#     )

#     assert (
#         len(criterion_subconfig["gradient_based_losses"])
#             == len(criterion_subconfig["gradient_based_losses_weights"])
#     ), "Number of weights should be equal to number of gradient based losses"

#     check_criterion_config(criterion_subconfig["task_loss"])

#     for gradient_based_loss_type \
#         in criterion_subconfig["gradient_based_losses"]:

#         if gradient_based_loss_type in [
#             ON_MANIFOLD_LOSS
#         ]:

#             check_gradient_based_loss(
#                 gradient_based_loss_type,
#                 criterion_subconfig[gradient_based_loss_type]
#             )


# def check_gradient_based_loss(
#     gradient_based_loss_type,
#     gradient_based_loss_config
# ):

#     if gradient_based_loss_type == ON_MANIFOLD_LOSS:
#         check_dict(
#             gradient_based_loss_config,
#             ["projector"],
#             check_reverse=True
#         )
#         check_projector_config(gradient_based_loss_config["projector"])
#     else:
#         raise_unknown(
#             "gradient_based_loss_type",
#             gradient_based_loss_type,
#             f"{DIVERSE_GRADIENTS_LOSS_KEY} config"
#         )


# def check_projector_config(projector_config):
#     check_dict(
#         projector_config,
#         ["type", "latent_dim", "unlabeled_data"],
#         check_reverse=True
#     )
#     check_projector_data_config(projector_config["unlabeled_data"])


# def check_projector_data_config(projector_data_config):
#     check_dict(
#         projector_data_config,
#         ["type", "total_number_of_images"],
#         optional_keys=[TINY_IMAGENET_TEST, "dsprites"],
#         check_reverse=True
#     )
#     data_type = projector_data_config["type"]
#     if data_type == TINY_IMAGENET_TEST:
#         check_config_for_type(
#             projector_data_config,
#             data_type,
#             check_tiny_imagenet_test_data_config
#         )
#     elif data_type == "dsprites":
#         check_config_for_type(
#             projector_data_config,
#             data_type,
#             check_dsprites_data_config
#         )
#     else:
#         raise_unknown(
#             "manifold projector data type",
#             data_type,
#             "manifold projector data config"
#         )


# def check_tiny_imagenet_test_data_config(tiny_imagenet_test_data_config):
#     check_dict(
#         tiny_imagenet_test_data_config,
#         ["normalize", "path"],
#         check_reverse=True
#     )
#     assert_type(tiny_imagenet_test_data_config["normalize"], bool)


# def check_statistics_config(statistics_config):
#     check_dict(
#         statistics_config,
#         [
#             "batchwise",
#             "keep_modelwise",
#             "use_wandb",
#             "use_tb"
#         ],
#         optional_keys=["wandb", "tb"],
#         check_reverse=True
#     )
#     assert_type(statistics_config["batchwise"], bool)
#     assert_type(statistics_config["keep_modelwise"], bool)
#     assert_type(statistics_config["use_wandb"], bool)
#     assert_type(statistics_config["use_tb"], bool)
#     if statistics_config["use_wandb"]:
#         check_config_for_type(
#             statistics_config,
#             "wandb",
#             check_wandb_config
#         )
#     if statistics_config["use_tb"]:
#         check_config_for_type(
#             statistics_config,
#             "tb",
#             check_tb_config
#         )


# def check_wandb_config(wandb_config, raise_if_wrong=True):

#     check_1 = False
#     check_2 = False
#     check_3 = False
#     use_at_least_one_stat = False

#     check_1 = check_dict(
#         wandb_config,
#         [
#             "netrc_path",
#             "stats"
#         ],
#         optional_keys=["wandb_init_kwargs"],
#         check_reverse=True,
#         raise_if_wrong=raise_if_wrong
#     )
#     if check_1:
#         wandb_stats_config = wandb_config["stats"]
#         check_2 = check_dict(
#             wandb_stats_config,
#             [
#                 "metrics",
#                 "loss",
#                 "input",
#                 "logits",
#                 "prediction",
#                 "target"
#             ],
#             check_reverse=True,
#             raise_if_wrong=raise_if_wrong
#         )
#         if check_2:
#             for to_use_stat in wandb_stats_config.values():
#                 check_3 = assert_type(
#                     to_use_stat,
#                     bool,
#                     raise_if_wrong=raise_if_wrong
#                 )
#                 if not check_3:
#                     break
#                 use_at_least_one_stat = use_at_least_one_stat or to_use_stat
#             if raise_if_wrong:
#                 assert use_at_least_one_stat, \
#                     "No stats are requested to be logged in wandb:\n{}".format(
#                         wandb_stats_config
#                     )

#     return check_1 and check_2 and check_3 and use_at_least_one_stat


# def check_tb_config(tb_config, raise_if_wrong=True):
#     check_1 = check_dict(
#         tb_config,
#         [
#             "credentials_path",
#             "upload_online"
#         ],
#         check_reverse=True,
#         raise_if_wrong=raise_if_wrong
#     )
#     if check_1:
#         check_2 = assert_type(
#             tb_config["upload_online"],
#             bool,
#             raise_if_wrong=raise_if_wrong
#         )
#     return check_1 and check_2


# def check_model_config(model_config):
#     check_dict(
#         model_config,
#         ["type"],
#         optional_keys=[
#             "resnet18",
#             "resnet50",
#             "linear",
#             REDNECK_ENSEMBLE_KEY,
#             "ensemble_weights",
#             "diverse_vit_switch_ensemble",
#             DIV_DIS_MODEL_NAME,
#             "mlp",
#             "timm_model",
#             "wrapper",
#             "torch_load",
#             "wrappers",
#             "separation_config"
#         ],
#         check_reverse=True
#     )
#     model_type = model_config["type"]
#     if model_type == "resnet50" or model_type == "resnet18":
#         check_config_for_type(
#             model_config,
#             model_type,
#             check_resnet_config
#         )
#     elif model_type == REDNECK_ENSEMBLE_KEY:
#         check_config_for_type(
#             model_config,
#             model_type,
#             check_redneck_ensemble_config
#         )
#     elif model_type == "diverse_vit_switch_ensemble":
#         pass
#     elif model_type == DIV_DIS_MODEL_NAME:
#         pass
#     elif model_type == ANY_KEY:
#         pass
#     elif model_type == "mlp":
#         pass
#     elif model_type == "timm_model":
#         pass
#     elif model_type == "torch_load":
#         pass
#     elif model_type == "linear":
#         pass
#     else:
#         raise_unknown(
#             "model type",
#             model_type,
#             "model config"
#         )


# def check_resnet_config(resnet_config):
#     check_dict(
#         resnet_config,
#         [
#             "pretrained",
#             "n_classes",
#             "n_channels"
#         ],
#         check_reverse=True
#     )
#     assert_type(resnet_config["pretrained"], bool)


# def check_redneck_ensemble_config(redneck_ensemble_config):
#     check_dict(
#         redneck_ensemble_config,
#         [
#             "base_estimator",
#             "n_models"
#         ],
#         optional_keys=[
#             "weights",
#             SINGLE_MODEL_KEY,
#             "identical",
#             "feature_extractor",
#             POE_KEY,
#             "random_select",
#             "keep_inactive_on_cpu",
#             "split_last_linear_layer",
#             "freeze_feature_extractor"
#         ],
#         check_reverse=True
#     )
#     check_model_config(redneck_ensemble_config["base_estimator"])


# def check_autogenerated_params(autogenerated_params_config):
#     check_dict(
#         autogenerated_params_config,
#         [
#             "input_csv",
#             OUTPUT_CSV_KEY
#         ],
#         check_reverse=True
#     )
#     check_csv_line_config(autogenerated_params_config["input_csv"])
#     check_csv_line_config(autogenerated_params_config[OUTPUT_CSV_KEY])


def check_csv_line_config(csv_config):
    check_dict(
        csv_config,
        [
            "path",
            "row_number",
            "spreadsheet_url",
            "worksheet_name"
        ],
        check_reverse=True
    )


# def assert_type(object_to_check, expected_type, raise_if_wrong=True):

#     same_type = type(object_to_check) == expected_type

#     if raise_if_wrong:
#         assert same_type, \
#             f"Type of {object_to_check} is expected to be: {expected_type}"

#     return same_type
