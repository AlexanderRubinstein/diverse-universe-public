from typing import (
    Dict,
    List,
    Tuple,
    Union,
    Any
)
import random
import os
import sys
# import torch.optim as optim
import torch
import wandb
# import torch.optim as optim
import numpy as np
import contextlib
# from torch.profiler import (
#     profile,
#     ProfilerActivity
# )
# import copy
from stuned.utility.utils import (
    NAME_SEP,
    get_project_root_path,
    log_or_print,
    get_device,
    update_dict_by_nested_key,
    get_with_assert,
    raise_unknown,
    apply_random_seed,
    save_checkpoint,
    remove_file_or_folder,
    append_dict,
    get_even_from_wrapped,
    dicts_with_non_intersecting_keys,
    remove_elements_from_the_end,
    read_checkpoint
)
from stuned.utility.logger import (
    INDENT,
    RedneckLogger,
    try_to_log_in_csv,
    try_to_log_in_wandb,
    make_logger,
    # make_base_estimator_name
    # LOGGING_CONFIG_KEY,
    # GDRIVE_FOLDER_KEY,
)
from stuned.utility.configs import (
    RUN_PATH_CONFIG_KEY
)
from stuned.local_datasets.imagenet1k import (
    IMAGENET_KEY,
    IMAGENET2012_CLASSES_LIST
)


# local modules
# sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, get_project_root_path())
# from utility.utils import (
#     NAME_SEP,
#     read_checkpoint,
#     log_or_print,
#     get_device,
#     raise_unknown,
#     append_dict,
#     save_checkpoint,
#     remove_elements_from_the_end,
#     dicts_with_non_intersecting_keys,
#     apply_random_seed,
#     pretty_json,
#     remove_file_or_folder,
#     get_with_assert,
#     get_even_from_wrapped,
#     update_dict_by_nested_key
# )
# from utility.logger import (
#     INDENT,
#     BASE_ESTIMATOR_LOG_SUFFIX,
#     RedneckLogger,
#     make_logger,
#     store_profiler_results,
#     dump_profiler_results,
#     try_to_log_in_csv,
#     make_base_estimator_name,
#     tb_log,
#     try_to_log_in_wandb
# )
# from train_eval.models import (
#     is_ensemble
# )
from diverse_universe.train.utils import (
    BASE_ESTIMATOR_LOG_SUFFIX,
    make_base_estimator_name
)
from diverse_universe.local_models.ensemble import (
    is_ensemble
)
# from diverse_universe.local_models.wrappers import (
#     wrap_model
# )
from diverse_universe.local_models.common import (
    build_model,
    # check_model,
    wrap_model
)
from diverse_universe.train.optimizer import (
    make_nan_checking_optimizer,
    make_optimizer,
    make_lr_scheduler
)
# from local_algorithms.div_dis import (
#     DIV_DIS_LOSS_NAMES,
#     DIV_DIS_Y_PRED_KEY,
#     DIV_DIS_UNLABELED_PRED_Y,
#     DIV_DIS_DEFAULT_LOGITS_PROCESSOR,
#     OBJECTIVE_KEY,
#     make_infinite_data_iterator,
#     make_nan_checking_optimizer
# )
# from local_algorithms.common import (
#     make_wilds_algorithm
# )
# from train_eval.losses import (
#     LOSS_STATISTICS_NAME,
#     INPUT_GRAD_NAME,
#     DivDisLossWrapper,
#     DiverseGradientsLoss,
#     make_criterion,
#     check_criterion,
#     requires_input_gradient
# )
from diverse_universe.train.losses import (
    LOSS_STATISTICS_NAME,
    DivDisLossWrapper,
    make_criterion
)
from diverse_universe.train.metrics import make_metric
# from train_eval.utils import (
#     make_optimizer,
#     make_lr_scheduler,
#     take_from_2d_tensor
# )
from diverse_universe.local_datasets.common import (
    EVAL_ON_TRAIN_LOGS_NAME,
    get_dataloaders
)
# from local_datasets.features_labeller import (
#     OFF_DIAG_COMMON_PREFIX,
#     DIAG_COMMON_PREFIX
# )
# from utility.configs import (
#     RUN_PATH_CONFIG_KEY
# )
from diverse_universe.train.configs import (
    DEFAULT_SETUP,
    # WILDS_SETUP
)
# from local_models.diverse_vit import is_diverse_vit_output
# from utility.imports import (
#     FROM_CLASS_KEY
# )
from diverse_universe.local_models.utils import (
    ModelClassesWrapper
)
# # TODO(Alex | 30.01.2024): Uncomment when dependency fixed
# # from external_libs.div_dis import (
# #     detach_and_clone,
# #     collate_list,
# #     process_outputs_functions
# # )
# from local_datasets.imagenet1k import (
#     IMAGENET_KEY,
#     IMAGENET2012_CLASSES_LIST
# )
sys.path.pop(0)


# # statistics
TRAIN_LOGS_NAME = "train data with updating weights"
PREDICTION_STAT_KEY = "prediction"
# GROUND_TRUTH_PROBS_STAT_KEY = "ground truth probs"
# DIVERSITY_MEASURE_STAT_KEY = "diversity measure"
MEAN_STAT_NAME = "mean"
MAX_STAT_NAME = "max"
MIN_STAT_NAME = "min"
STD_STAT_NAME = "std"
AGGREGATED_STAT_NAMES = (
    MEAN_STAT_NAME,
    MAX_STAT_NAME,
    MIN_STAT_NAME,
    STD_STAT_NAME
)
NESTED_LOGS_SEPARATOR = "->"
AFTER_MEAN_SEP = " +- "
CONFIG_LOG_NAME = "Config"
# ALGORITHM_KEY = "algorithm"
NUM_CLASSES_KEYS = ("num_classes", "num_classes_per_feature")


# wandb
STAT_WANDB_INPUT = "input"
STAT_WANDB_LOGITS = "logits"
STAT_WANDB_PREDICTION = "prediction"
STAT_WANDB_TARGET = "target"
# MEDIA_STAT_NAMES = [STAT_WANDB_INPUT, INPUT_GRAD_NAME]
WANDB_ONLY_STATS = [
    STAT_WANDB_INPUT,
    STAT_WANDB_LOGITS,
    STAT_WANDB_PREDICTION,
    STAT_WANDB_TARGET
]
LATEST_CHECKPOINT = "latest_checkpoint.pkl"
MODEL_KEY = "model"
EXP_NAME_KEY = "experiment_name"
CHECKPOINT_TEMPLATE = (
    (EXP_NAME_KEY, None),
    ("current_epoch", 0)
)
# SHORT_BASE_ESTIMATOR_LOG_SUFFIX = BASE_ESTIMATOR_LOG_SUFFIX.replace(NAME_SEP, '')
SCIENTIFIC_NOTATION_THRESHOLD = 1e-4
LOG_N_TIMES = 10
# BASE_ESTIMATOR_LOG_SUFFIX = "base_estimator"


def train_eval_loop(
    experiment_config,
    logger=None,
    processes_to_kill_before_exiting=[]
):
    params_config = experiment_config["params"]
    to_train = params_config["to_train"]
    to_eval = params_config["to_eval"]

    # TODO(Alex | 26.01.2024): move this to stuned
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "Not found")
    # logger.log("slurm_job_id: {}".format(slurm_job_id))
    log_or_print("slurm_job_id: {}".format(slurm_job_id), logger)
    try_to_log_in_csv(
        logger,
        "slurm job id",
        slurm_job_id
    )

    checkpoint = get_checkpoint(experiment_config, logger=logger)

    ensemble_weights = experiment_config[MODEL_KEY].get("ensemble_weights")
    if ensemble_weights is not None:
        model = checkpoint[MODEL_KEY]
        assert is_ensemble(model)
        model.set_weights(ensemble_weights)

    device = get_device(params_config["use_gpu"])

    if to_train:
        train_config = params_config["train"]
        freeze_model_on_first_epoch \
            = train_config["freeze_model_on_first_epoch"]
        if train_config.get("reset_epochs", False):
            checkpoint["current_epoch"] = 0
        starting_epoch = checkpoint["current_epoch"]
        total_n_epochs = int(train_config["n_epochs"])
        assert total_n_epochs > starting_epoch
        type_of_run = "train/eval" if to_eval else "train"
        checkpoints_to_save = train_config.get("checkpoints_to_save", [])
    elif not to_eval:
        raise Exception(
            "At least one of \"to_train\" or \"to_eval\" should be True."
        )

    if to_eval:
        eval_only_last_epoch = params_config["eval"]["eval_only_last_epoch"]
        if not to_train:
            type_of_run = "eval"
            starting_epoch = 0
            total_n_epochs = 1

    wandb_run = logger.wandb_run
    tb_run = logger.tb_run

    # TODO(Alex |26.01.2024): make it more parametrizible and less hardcoded
    # e.g. give nested keys inside wandb.config
    if experiment_config.get("wandb_sweep", False):

        def patch_params(config, nested_key, value, logger, name):
            # logger.log(f"Taking {name} = {value} from wandb.config")
            log_or_print(f"Taking {name} = {value} from wandb.config", logger)
            update_dict_by_nested_key(
                experiment_config,
                nested_key,
                value
            )

        wandb_config = wandb.config
        lr_nested_key = ["params", "train", "optimizer", "start_lr"]
        opt_type_nested_key = ["params", "train", "optimizer", "type"]
        patch_params(
            experiment_config,
            lr_nested_key,
            wandb_config.learning_rate,
            logger,
            "lr"
        )
        patch_params(
            experiment_config,
            opt_type_nested_key,
            wandb_config.optimizer,
            logger,
            "optimizer"
        )

    statistics_config = experiment_config["statistics"]

    keep_modelwise = statistics_config["keep_modelwise"]

    # use_tb = statistics_config["use_tb"]

    train_dataloader, val_dataloaders, unlabeled_dataloaders = get_dataloaders(
        experiment_config,
        logger=logger
    )

    if unlabeled_dataloaders is None:
        unlabeled_iterator = None
    else:
        assert len(unlabeled_dataloaders) == 1
        unlabeled_iterator = make_infinite_data_iterator(
            next(iter(unlabeled_dataloaders.values()))
        )

    # logger.log(
    #     "Start {} loop".format(type_of_run) + (
    #         " from epoch {} to epoch {}.".format(
    #             starting_epoch + 1,
    #             total_n_epochs
    #         )
    #             if to_train
    #             else ""
    #     )
    # )
    log_or_print(
        "Start {} loop".format(type_of_run) + (
            " from epoch {} to epoch {}.".format(
                starting_epoch + 1,
                total_n_epochs
            )
                if to_train
                else ""
        ),
        logger
    )

    # if use_tb:
    #     tb_run.add_text(
    #         CONFIG_LOG_NAME,
    #         pretty_json(experiment_config),
    #         global_step=0
    #     )

    setup = params_config.get("setup", DEFAULT_SETUP)
    cache_path = experiment_config["cache_path"]

    experiment_config["model"]["d_out"] \
        = infer_num_clases(experiment_config["data"])

    if setup == DEFAULT_SETUP:
        prepare_setup_default(
            experiment_config,
            checkpoint,
            cache_path,
            device,
            logger
        )
    # elif setup == WILDS_SETUP:

    #     assert to_train, \
    #         f"Should always have to_train == True if setup == {WILDS_SETUP}"
    #     train_config = get_with_assert(params_config, "train")
    #     prepare_setup_wilds(
    #         params_config,
    #         get_with_assert(experiment_config, "model"),
    #         get_with_assert(train_config, "criterion"),
    #         checkpoint,
    #         train_dataloader,
    #         device
    #     )
    else:
        raise_unknown("setup for preparing setup", setup, "run_epoch")

    move_to_device(checkpoint, device)

    compute_metrics = make_metric(params_config, device)

    # if compute_metrics.final_aggregation is not None:
    #     if compute_metrics.aggregatable_stages is None:
    #         stages_to_aggregate = "all stages"
    #     else:
    #         stages_to_aggregate = str(compute_metrics.aggregatable_stages)

    checkpoint["compute_metrics"] = compute_metrics

    # csv_column_maker = get_csv_column_name_maker(
    #     experiment_config["data"]["dataset"]["type"]
    # )

    csv_column_maker = make_csv_column_name_default

    best_aggregated_metric = None

    all_stats = {}

    for epoch in range(starting_epoch, total_n_epochs):

        apply_random_seed(
            get_epoch_random_seed(
                params_config["random_seed"],
                epoch,
                freeze_model_on_first_epoch if to_train else False
            )
        )

        if to_train:

            # logger.log(
            #     "Epoch {}/{}.".format(epoch + 1, total_n_epochs)
            # )

            log_or_print(
                "Epoch {}/{}.".format(epoch + 1, total_n_epochs),
                logger
            )

            train_stage_log_msg = "Train stage for epoch {}".format(
                epoch + 1
            )

            freeze_model = (
                epoch == 0
                    and freeze_model_on_first_epoch
            )

            if freeze_model:
                train_stage_log_msg += " (model is frozen)"

            # logger.log(train_stage_log_msg)
            log_or_print(train_stage_log_msg, logger)

            all_stats[TRAIN_LOGS_NAME] = run_epoch(
                experiment_config,
                train_dataloader,
                checkpoint,
                device,
                wandb_run,
                tb_run,
                epoch,
                is_train=True,
                stage_name=TRAIN_LOGS_NAME,
                logger=logger,
                freeze_model=freeze_model,
                unlabeled_iterator=unlabeled_iterator
            )
            # TODO(Alex | 31.10.2023): avoid copying; e.g. load from state_dict
            # in prepare_for_unpickling for each torch.nn.Module in checkpoint
            run_folder = experiment_config[RUN_PATH_CONFIG_KEY]
            checkpoint_to_save = (
                # copy.deepcopy(checkpoint)
                #     if setup == WILDS_SETUP
                #         else checkpoint
                checkpoint
            )
            save_checkpoint(
                checkpoint_to_save,
                run_folder,
                checkpoint_name=LATEST_CHECKPOINT,
                logger=logger
            )
            if epoch + 1 in checkpoints_to_save:
                intermediate_checkpoint_path = save_checkpoint(
                    checkpoint_to_save,
                    run_folder,
                    checkpoint_name=f"checkpoint_{epoch + 1}.pkl",
                    logger=logger
                )
                try_to_log_in_csv(
                    logger,
                    f"checkpoint {epoch + 1}",
                    intermediate_checkpoint_path
                )

        if to_eval:

            if (
                not eval_only_last_epoch
                    or epoch + 1 == total_n_epochs
            ):
                # logger.log(
                #     "Validation stage" + (
                #         " for epoch {}".format(epoch + 1)
                #             if to_train
                #             else ""
                #     )
                # )
                log_or_print(
                    "Validation stage" + (
                        " for epoch {}".format(epoch + 1)
                            if to_train
                            else ""
                    ),
                    logger
                )

                for val_dataset_name in val_dataloaders.keys():
                    all_stats[val_dataset_name] \
                        = run_epoch(
                            experiment_config,
                            val_dataloaders[val_dataset_name],
                            checkpoint,
                            device,
                            wandb_run,
                            tb_run,
                            epoch,
                            is_train=False,
                            stage_name=val_dataset_name,
                            logger=logger
                        )

        assert len(all_stats) > 0

        # if not compute_metrics.report_best_metric:
        #     to_log_in_csv = False
        # else:
        to_log_in_csv = True

        # if compute_metrics.final_aggregation is not None:

        #     if compute_metrics.aggregatable_stages is not None:
        #         aggregatable_stats = {}
        #         for stage_name in compute_metrics.aggregatable_stages:
        #             stage_stats = get_with_assert(all_stats, stage_name)
        #             aggregatable_stats[stage_name] = stage_stats
        #     else:
        #         aggregatable_stats = all_stats

        #     aggregated_metric = compute_metrics.aggregate(aggregatable_stats)

        #     all_stats["overall"] = {}
        #     overall_stats = all_stats["overall"]

        #     log_or_print(
        #         f"Aggregated metric for {stages_to_aggregate}"
        #         f" in epoch {epoch + 1}:\n\n",
        #         logger
        #     )

        #     for aggregated_metric_name, aggregated_metric_value \
        #         in aggregated_metric.items():

        #         log_or_print(
        #             f"{aggregated_metric_name}: {aggregated_metric_value}\n",
        #             logger
        #         )

        #         overall_stats[aggregated_metric_name] \
        #             = str(aggregated_metric_value)

        #     try_to_log_in_wandb(
        #         logger,
        #         aggregated_metric,
        #         step=epoch
        #     )

        #     logger.log_separator()

        #     if compute_metrics.report_best_metric:

        #         main_aggregated_metric_name = get_with_assert(
        #             aggregated_metric,
        #             compute_metrics.final_aggregation
        #         )

        #         to_log_in_csv = False

        #         if (
        #                 best_aggregated_metric is None
        #             or
        #                 best_aggregated_metric < main_aggregated_metric_name
        #         ):
        #             to_log_in_csv = True
        #             best_aggregated_metric = main_aggregated_metric_name

        log_stats_in_csv(
            all_stats,
            keep_modelwise,
            to_log_in_csv,
            csv_column_maker,
            logger
        )

    if to_train:
        final_checkpoint_path = save_checkpoint(
            checkpoint,
            experiment_config[RUN_PATH_CONFIG_KEY],
            logger=logger
        )
        checkpoint_symlink = experiment_config.get("checkpoint_symlink")
        if (
                checkpoint_symlink is not None
            and
                not os.path.exists(checkpoint_symlink)
        ):
            os.makedirs(
                os.path.dirname(checkpoint_symlink),
                exist_ok=True
            )
            os.symlink(final_checkpoint_path, checkpoint_symlink)
        try_to_log_in_csv(
            logger,
            "final checkpoint path",
            final_checkpoint_path
        )
        latest_checkpoint_path = os.path.join(
            experiment_config[RUN_PATH_CONFIG_KEY],
            LATEST_CHECKPOINT
        )
        if os.path.exists(latest_checkpoint_path):
            remove_file_or_folder(latest_checkpoint_path)

    # logger.log("Finished {} loop.".format(type_of_run))
    log_or_print("Finished {} loop.".format(type_of_run), logger)


def get_epoch_random_seed(base_random_seed, epoch, freeze_model_on_first_epoch):
    epoch_random_seed = base_random_seed + epoch
    if epoch > 0 and freeze_model_on_first_epoch:
        epoch_random_seed -= 1
    return epoch_random_seed


def move_to_device(checkpoint, device):

    # if "algorithm" in checkpoint:
    #     checkpoint["algorithm"].model.to(device)

    if "model" in checkpoint:
        checkpoint["model"].to(device)


def infer_num_clases(data_config):
    dataset_config = get_with_assert(data_config, "dataset")
    dataset_type = get_with_assert(dataset_config, "type")
    if dataset_type == IMAGENET_KEY:
        return len(IMAGENET2012_CLASSES_LIST)
    specific_dataset_config = get_with_assert(dataset_config, dataset_type)
    for key in NUM_CLASSES_KEYS:
        if key in specific_dataset_config:
            return specific_dataset_config[key]
    return None


# based on: https://github.com/zhangchbin/OnlineLabelSmoothing/blob/7eaf70c8da7c68ba2170cbd88e0d918e86fa3f14/cifar/scripts/loss_all_methods.py#L31
def smooth_labels(label_one_hot, smoothing_eps):
    num_classes = label_one_hot.size(-1)
    return (
        label_one_hot * (1. - smoothing_eps) + smoothing_eps / float(num_classes)
    )


def run_epoch(
    experiment_config,
    dataloader,
    checkpoint,
    device,
    wandb_run,
    tb_run,
    epoch=0,
    is_train=False,
    stage_name="",
    logger=None,
    freeze_model=False,
    unlabeled_iterator=None
):

    def update_epoch_stats(
        stage_name,
        epoch_stats,
        batch_stats,
        metrics_names,
        losses_names
    ):

        batch_stats_for_stage_name = batch_stats[stage_name]

        stats_to_extract = metrics_names + losses_names

        # if GROUND_TRUTH_PROBS_STAT_KEY in batch_stats_for_stage_name:
        #     stats_to_extract += [GROUND_TRUTH_PROBS_STAT_KEY]

        append_dict(
            epoch_stats[stage_name],
            {
                stat_name: batch_stats_for_stage_name[stat_name]
                    for stat_name in stats_to_extract
            }
        )
        # if GROUND_TRUTH_PROBS_STAT_KEY in batch_stats_for_stage_name:
        #     batch_stats_for_stage_name.pop(GROUND_TRUTH_PROBS_STAT_KEY)

    def extract_inputs_targets_and_move_to_device(
        dataloader_items,
        label_names,
        device,
        has_metadata
    ):
        if has_metadata:
            metadata = dataloader_items[-1]
            dataloader_items = dataloader_items[:-1]
        else:
            metadata = None
        if label_names is not None:
            assert len(dataloader_items) > 2
            inputs = dataloader_items[0]
            targets = {
                label_name: target.to(device)
                    for label_name, target
                        in zip(label_names, dataloader_items[1:])
            }
        else:
            assert len(dataloader_items) == 2
            inputs, targets = dataloader_items
            targets = targets.to(device)

        inputs = inputs.to(device)

        return inputs, targets, metadata

    def usage_info(object, logger, name):
        if object is not None:
            # logger.log("Using: {} {}".format(
            #     type(object).__name__,
            #     name
            # ))
            log_or_print(
                "Using: {} {}".format(
                    type(object).__name__,
                    name
                ),
                logger
            )

    if logger is None:
        logger = make_logger()

    # logger.log("Running \"{}\" for epoch {}".format(stage_name, epoch + 1))
    log_or_print(
        "Running \"{}\" for epoch {}".format(stage_name, epoch + 1),
        logger
    )


    params_config = experiment_config["params"]

    setup = params_config.get("setup", DEFAULT_SETUP)

    epoch_stats = {stage_name: {}}

    statistics_config = experiment_config["statistics"]
    use_wandb = statistics_config["use_wandb"]
    use_tb = statistics_config["use_tb"]
    batchwise_statistics = statistics_config["batchwise"]

    if use_wandb:
        wandb_stats_config = statistics_config["wandb"]["stats"]
        assert_wandb(wandb_run, batchwise_statistics, statistics_config)

    select_train_eval(setup, checkpoint, is_train)

    compute_metrics = checkpoint["compute_metrics"]

    if hasattr(dataloader, "label_names"):
        label_names = dataloader.label_names
    else:
        label_names = None

    total_num_batches = len(dataloader)
    assert total_num_batches, "Empty dataloader."
    randomly_sampled_batch_id = random.randint(0, total_num_batches - 1)

    has_metadata = get_even_from_wrapped(
        dataloader,
        "dataloader",
        "has_metadata"
    )

    # if setup == WILDS_SETUP:

    #     algorithm = checkpoint["algorithm"]

    #     if has_metadata:
    #         epoch_y_pred = []
    #         epoch_y_true = []
    #         epoch_metadata = []

    #     gradient_accumulation_steps \
    #         = algorithm.config.gradient_accumulation_steps
    #     log_every = algorithm.config.log_every

    # at_least_one_off_diag = False
    if is_train or stage_name == EVAL_ON_TRAIN_LOGS_NAME:
        train_config = get_with_assert(params_config, "train")
        off_diag_supervision = train_config.get("off_diag_supervision")

    if setup == DEFAULT_SETUP:
        model = checkpoint["model"]

        model_config = get_with_assert(experiment_config, "model")
        wrappers = model_config.get("wrappers")
        if wrappers is not None:
            model = wrap_model(model, wrappers.get(stage_name))

        optimizer = checkpoint.get("optimizer")
        usage_info(optimizer, logger, "optimizer")
        # logger.log("Current learning rate: {}".format(
        #     optimizer.param_groups[0]['lr']
        # ))
        log_or_print(
            "Current learning rate: {}".format(
                optimizer.param_groups[0]['lr']
            ),
            logger
        )
        lr_scheduler = checkpoint.get("lr_scheduler")
        usage_info(lr_scheduler, logger, "lr_scheduler")
        criterion = checkpoint.get("criterion")

    if is_train:
        log_batch_stats_every = len(dataloader) // LOG_N_TIMES + 1
    else:
        log_batch_stats_every = 1
    for batch_idx, dataloader_items in enumerate(dataloader):

        log_this_batch = (batch_idx % log_batch_stats_every == 0)

        if is_train or stage_name == EVAL_ON_TRAIN_LOGS_NAME:
            # if off_diag_supervision is not None and len(dataloader_items) > 2:
            #     at_least_one_off_diag = True
            #     all_labels = dataloader_items[1:]
            #     if off_diag_supervision == "clean":
            #         label_id = 0
            #     elif off_diag_supervision == "mixed":
            #         model_config = get_with_assert(experiment_config, "model")
            #         num_classes = get_with_assert(model_config, "d_out")
            #         label_id = get_mixed_label_id(
            #             all_labels,
            #             params_config["random_seed"],
            #             num_classes
            #         )
            #     else:
            #         raise_unknown(
            #             "off_diag_supervision",
            #             off_diag_supervision,
            #             "extract_inputs_targets_and_move_to_device"
            #         )

            #     inputs = dataloader_items[0]

            #     targets = take_from_2d_tensor(
            #         torch.cat(all_labels).view(
            #             len(all_labels),
            #             0
            #         ),
            #         label_id,
            #         dim=-1
            #     )

            #     dataloader_items = [inputs, targets]

            # else:
            #     assert len(dataloader_items) == 2, \
            #         "Looks like off_diag_percent is set for dataset " \
            #         "while off_diag_supervision is not set in train config."

            assert len(dataloader_items) == 2

        inputs, targets, metadata = extract_inputs_targets_and_move_to_device(
            dataloader_items,
            label_names,
            device,
            has_metadata
        )

        # # start of profiler context
        # with (
        #     profile(
        #         activities=[ProfilerActivity.CPU]
        #             + ([ProfilerActivity.CUDA] if device != "cpu" else []),
        #         profile_memory=True,
        #         record_shapes=True
        #     ) if params_config["to_profile"] else contextlib.nullcontext()
        # ) as prof:

        # TODO(Alex | 25.07.2024): remove this context
        with contextlib.nullcontext():

            logger.progress(
                "Processing batch",
                batch_idx + 1,
                total_num_batches
            )
            inputs = inputs.to(device)

            if setup == DEFAULT_SETUP:
                if unlabeled_iterator is not None:
                    unlabeled_batch = next(unlabeled_iterator)
                else:
                    unlabeled_batch = None
                (
                    metrics,
                    outputs,
                    train_info
                ) \
                    = process_batch_default(
                        model=model,
                        inputs=inputs,
                        targets=targets,
                        is_train=is_train,
                        freeze_model=freeze_model,
                        criterion=criterion,
                        optimizer=(optimizer if is_train else None),
                        compute_metrics=compute_metrics,
                        logger=logger,
                        stage_name=stage_name,
                        unlabeled_batch=unlabeled_batch,
                        log_this_batch=log_this_batch
                    )
            # elif setup == WILDS_SETUP:

            #     unlabeled_batch = None
            #     unlabeled_metadata = None

            #     if is_train:
            #         assert unlabeled_iterator is not None
            #         unlabeled_batch = next(unlabeled_iterator)
            #         if isinstance(unlabeled_batch, list):
            #             if has_metadata:
            #                 unlabeled_metadata = unlabeled_batch[-1]
            #             unlabeled_batch = unlabeled_batch[0]
            #         else:
            #             assert torch.is_tensor(unlabeled_batch)
            #             assert not has_metadata

            #     (
            #         metrics,
            #         outputs,
            #         train_info,
            #         batch_results
            #     ) \
            #         = process_batch_wilds_algorithm(
            #             algorithm=algorithm,
            #             compute_metrics=compute_metrics,
            #             batch_idx=batch_idx,
            #             inputs=inputs,
            #             targets=targets,
            #             metadata=metadata,
            #             unlabeled_batch=unlabeled_batch,
            #             unlabeled_metadata=unlabeled_metadata,
            #             is_train=is_train,
            #             freeze_model=freeze_model,
            #             gradient_accumulation_steps=gradient_accumulation_steps,
            #             log_every=log_every,
            #             is_epoch_end=(batch_idx + 1 == total_num_batches),
            #             stage_name=stage_name,
            #             logger=logger
            #         )
            #     if has_metadata:
            #         epoch_y_true.append(detach_and_clone(batch_results["y_true"]))
            #         y_pred = detach_and_clone(batch_results[DIV_DIS_Y_PRED_KEY])
            #         y_pred = process_outputs_functions\
            #             [DIV_DIS_DEFAULT_LOGITS_PROCESSOR](y_pred)
            #         epoch_y_pred.append(y_pred)
            #         epoch_metadata.append(
            #             detach_and_clone(batch_results["metadata"])
            #         )
            else:
                raise_unknown("setup for batch processing", setup, "run_epoch")

            if (
                    not batchwise_statistics
                and
                    log_this_batch
            ):
                (
                    batch_stats,
                    batch_media,
                    metrics_names,
                    losses_names
                ) \
                    = make_batch_stats(
                        experiment_config,
                        metrics,
                        inputs,
                        outputs,
                        targets,
                        train_info,
                        batch_idx,
                        epoch,
                        stage_name
                    )
            else:
                batch_stats = None

            if (
                not batchwise_statistics
                    and batch_idx == randomly_sampled_batch_id
            ):
                epoch_media = batch_media

            if batch_stats is not None:
                if is_train:
                    assert len(losses_names) > 0
                update_epoch_stats(
                    stage_name,
                    epoch_stats,
                    batch_stats,
                    metrics_names,
                    losses_names
                )

            if batchwise_statistics:
                assert batch_stats is not None
                if use_wandb:
                    log_stats_in_wandb(
                        logger,
                        stage_name,
                        batch_stats,
                        batch_media,
                        metrics_names,
                        losses_names,
                        wandb_stats_config["metrics"],
                        wandb_stats_config["loss"],
                        step=(batch_idx + epoch * total_num_batches)
                    )
                # if use_tb:
                #     tb_log(
                #         tb_run,
                #         batch_stats,
                #         current_step=batch_idx,
                #         step_offset=epoch * total_num_batches,
                #         skip_key_func=is_wandb_only_stat
                #     )

        # # end of profiler context
        # if params_config["to_profile"]:
        #     store_profiler_results(logger, prof)

    # if GROUND_TRUTH_PROBS_STAT_KEY in epoch_stats[stage_name]:

    #     ground_truth_probs = epoch_stats[stage_name].pop(
    #         GROUND_TRUTH_PROBS_STAT_KEY
    #     )

    #     diversity_measure_for_all_targets = {
    #         target_name: compute_diversity_measure(
    #             ground_truth_probs_per_target
    #         )
    #             for target_name, ground_truth_probs_per_target
    #                 in ground_truth_probs.items()
    #     }
    #     diversity_measure_for_all_targets["mean"] = np.mean(
    #         np.array(list(diversity_measure_for_all_targets.values()))
    #     )

    # else:
    #     diversity_measure_for_all_targets = None

    # if (
    #         is_train
    #     and
    #         off_diag_supervision is not None
    # ):
    #     # for stage_name == EVAL_ON_TRAIN_LOGS_NAME we don't require this
    #     assert at_least_one_off_diag, \
    #         "Looks like off_diag_percent is not set for dataset " \
    #         "while off_diag_supervision is set in train config."

    # finish epoch
    epoch_stats = aggregate_stats(
        epoch_stats,
        only_mean=(not batchwise_statistics)
    )

    # if setup == WILDS_SETUP and has_metadata:
    #     dataset = dataloader.dataset
    #     epoch_y_pred = collate_list(epoch_y_pred)
    #     epoch_y_true = collate_list(epoch_y_true)
    #     epoch_metadata = collate_list(epoch_metadata)
    #     _, results_str = dataset.eval(
    #         epoch_y_pred, epoch_y_true, epoch_metadata
    #     )
    #     logger.log(results_str)

    if not batchwise_statistics and use_wandb:

        log_stats_in_wandb(
            logger,
            stage_name,
            epoch_stats,
            epoch_media,
            metrics_names,
            losses_names,
            wandb_stats_config["metrics"],
            wandb_stats_config["loss"],
            step=epoch
        )

    # if diversity_measure_for_all_targets is not None:

    #     epoch_stats[stage_name][DIVERSITY_MEASURE_STAT_KEY] \
    #         = diversity_measure_for_all_targets

    # if not batchwise_statistics and use_tb:

    #     tb_log(
    #         tb_run,
    #         epoch_stats,
    #         epoch,
    #         flush=True,
    #         skip_key_func=is_wandb_only_stat
    #     )

    flattened_epoch_stats = log_stats(
        logger,
        epoch_stats,
        stage_name,
        epoch=epoch,
    )

    if is_train and not freeze_model:

        if setup == DEFAULT_SETUP:
            make_scheduler_step(lr_scheduler, epoch_stats[stage_name])
        else:
            raise_unknown(
                "setup for making scheduler step",
                setup,
                "run_epoch"
            )

    checkpoint["current_epoch"] = epoch

    # if params_config["to_profile"]:
    #     dump_profiler_results(logger)

    after_epoch(checkpoint, setup, is_train, logger, freeze_model)

    return flattened_epoch_stats


# def get_mixed_label_id(all_labels, random_seed, num_classes):
#     base = 1
#     fingerprint = 0
#     for label in all_labels:
#         fingerprint += base * label
#         base *= num_classes
#     return (fingerprint + random_seed) % len(all_labels)


def increase_epoch_for_criterion(criterion):

    def try_to_increase_epoch(criterion):
        if hasattr(criterion, "increase_epoch"):
            criterion.increase_epoch()

    if isinstance(criterion, tuple):
        assert len(criterion) == 2
        try_to_increase_epoch(criterion[0])
        try_to_increase_epoch(criterion[1])
    else:
        try_to_increase_epoch(criterion)


def after_epoch(checkpoint, setup, is_train, logger, freeze_model):

    if not freeze_model:
        if setup == DEFAULT_SETUP:
            model = get_with_assert(checkpoint, "model")
            if hasattr(model, "after_epoch"):
                model.after_epoch(is_train, logger)
        if isinstance(model, ModelClassesWrapper):
            model = model.get_inner_object()
        if (
                "criterion" in checkpoint
            and
                is_train
        ):
            increase_epoch_for_criterion(checkpoint["criterion"])


# def check_algorithm(algorithm, expected_algorithm):
#     pass


def select_train_eval(setup, checkpoint, is_train):

    if setup == DEFAULT_SETUP:
        model = checkpoint["model"]
    # elif setup == WILDS_SETUP:
    #     model = checkpoint["algorithm"]
    else:
        raise_unknown(
            "setup for select_train_eval",
            setup,
            "select_train_eval"
        )

    if is_train:
        model.train()
    else:
        model.eval()


# def prepare_setup_wilds(
#     params_config,
#     model_config,
#     criterion_config,
#     checkpoint,
#     train_dataloader,
#     device
# ):
#     total_train_batches = len(train_dataloader)

#     grouper = get_even_from_wrapped(train_dataloader, "dataloader", "grouper")

#     algorithm_config = get_with_assert(params_config, ALGORITHM_KEY)

#     model = None
#     optimizer_config = None
#     scheduler_config = None

#     if algorithm_config.get("use_custom_model", False):

#         model = build_model(model_config)
#         train_config = get_with_assert(params_config, "train")
#         optimizer_config = get_with_assert(train_config, "optimizer")
#         scheduler_config = None
#         if "lr_scheduler" in optimizer_config:
#             scheduler_config = optimizer_config["lr_scheduler"]

#     algorithm = make_or_get_from_checkpoint(
#         "algorithm",
#         checkpoint,
#         params_config,
#         check_algorithm,
#         make_wilds_algorithm,
#         model_config=model_config,
#         criterion_config=criterion_config,
#         total_train_batches=total_train_batches,
#         grouper=grouper,
#         model=model,
#         optimizer_config=optimizer_config,
#         scheduler_config=scheduler_config,
#         device=device
#     )
#     return algorithm


def prepare_setup_default(
    experiment_config,
    checkpoint,
    cache_path,
    device,
    logger
):

    params_config = experiment_config["params"]

    model = make_or_get_from_checkpoint(
        MODEL_KEY,
        checkpoint,
        experiment_config,
        None,
        build_model,
        logger=logger
    )

    logger.log(
        "Moving model to device: {}".format(device)
    )

    if params_config["to_train"]:

        train_config = params_config["train"]

        optimizer_config = get_with_assert(train_config, "optimizer")

        if optimizer_config.get("nan_checking", False):
            wrapping_function = make_nan_checking_optimizer
        else:
            wrapping_function = None

        optimizer = make_or_get_from_checkpoint(
            "optimizer",
            checkpoint,
            train_config,
            None,
            make_optimizer,
            param_container=model,
            wrapping_function=wrapping_function
        )

        if "lr_scheduler" in optimizer_config:
            lr_scheduler = make_or_get_from_checkpoint(
                "lr_scheduler",
                checkpoint,
                optimizer_config,
                None,
                make_lr_scheduler,
                optimizer=optimizer,
                max_epochs=train_config["n_epochs"]
            )

        criterion = make_or_get_from_checkpoint(
            "criterion",
            checkpoint,
            train_config,
            None,
            make_criterion,
            cache_path=cache_path,
            logger=logger,
            device=device
        )
        if "unlabeled_criterion" in train_config:
            logger.log(
                f"Making unlabeled criterion"
            )
            unlabeled_criterion = make_criterion(
                train_config["unlabeled_criterion"],
                logger=logger
            )
            if (
                    isinstance(unlabeled_criterion, DivDisLossWrapper)
                and
                    unlabeled_criterion.weight is None
            ):
                logger.log(
                    f"Setting weight of unlabeled criterion equal "
                    f"to labeled criterion's weight: {criterion.weight}."
                )
                assert isinstance(criterion, DivDisLossWrapper)
                unlabeled_criterion.weight = criterion.weight
            checkpoint["criterion"] = tuple([criterion, unlabeled_criterion])


# def process_batch_wilds_algorithm(
#     algorithm,
#     compute_metrics,
#     batch_idx,
#     inputs,
#     targets,
#     metadata,
#     unlabeled_batch,
#     unlabeled_metadata,
#     is_train,
#     freeze_model,
#     gradient_accumulation_steps,
#     log_every,
#     is_epoch_end,
#     stage_name,
#     logger
# ):

#     def ensure_metadata(batch, metadata):

#         if metadata is None:
#             return torch.zeros((batch.shape[0], 1))

#         return metadata

#     if freeze_model:
#         frozen_optimizer = algorithm.algorithm.optimizer
#         frozen_schedulers = algorithm.algorithm.schedulers
#         algorithm.algorithm.optimizer = DummyOptimizer()
#         algorithm.algorithm.schedulers = []
#     else:
#         frozen_optimizer = None
#         frozen_schedulers = None

#     metadata = ensure_metadata(inputs, metadata)

#     labeled_batch = (inputs, targets, metadata)

#     if is_train:
#         if unlabeled_batch is not None:

#             unlabeled_metadata = ensure_metadata(
#                 unlabeled_batch,
#                 unlabeled_metadata
#             )
#             unlabeled_batch = (unlabeled_batch, unlabeled_metadata)
#             batch_results = algorithm.update(
#                 labeled_batch,
#                 unlabeled_batch,
#                 is_epoch_end=is_epoch_end
#             )
#         else:
#             batch_results = algorithm.update(
#                 labeled_batch,
#                 is_epoch_end=is_epoch_end
#             )
#         effective_batch_idx = (batch_idx + 1) / gradient_accumulation_steps
#         if algorithm.has_log and effective_batch_idx % log_every == 0:
#             logger.log(algorithm.get_pretty_log_str())
#             algorithm.reset_log()

#     else:
#         batch_results = algorithm.evaluate(labeled_batch)
#         intermediate_results = algorithm.process_batch(labeled_batch, None)
#         batch_results["objective"] \
#             = algorithm.objective(intermediate_results).item()

#     if frozen_optimizer is not None:
#         algorithm.algorithm.optimizer = frozen_optimizer

#     if frozen_schedulers is not None:
#         algorithm.algorithm.schedulers = frozen_schedulers

#     outputs = [[inputs, batch_results[DIV_DIS_Y_PRED_KEY].to(targets.device)]]

#     for key in batch_results.keys():
#         if (
#                 DIV_DIS_Y_PRED_KEY in key
#             and
#                 key not in (DIV_DIS_UNLABELED_PRED_Y, DIV_DIS_Y_PRED_KEY)
#         ):
#             outputs.append([inputs, batch_results[key].to(targets.device)])

#     metrics = make_metrics_dict(
#         compute_metrics,
#         targets,
#         outputs,
#         is_train
#     )

#     train_info_for_stage = {}
#     train_info = {stage_name: train_info_for_stage}

#     for loss_name in DIV_DIS_LOSS_NAMES:
#         extracted_loss = get_with_assert(batch_results, loss_name)
#         if loss_name == OBJECTIVE_KEY:
#             loss_name = "total_loss"

#         train_info_for_stage[loss_name] = extracted_loss

#     return (
#         metrics,
#         outputs,
#         train_info,
#         batch_results
#     )


# class DummyOptimizer(torch.nn.Module):

#     def step(self):
#         super().__init__()

#     def zero_grad(self):
#         pass


def process_batch_default(
    model,
    inputs,
    targets,
    is_train,
    freeze_model,
    criterion,
    optimizer,
    compute_metrics,
    logger,
    stage_name,
    unlabeled_batch,
    log_this_batch
):

    def overwrite_keys(stats_dict, suffix):
        return {
            key + suffix: value
                for key, value in stats_dict.items()
        }

    # if requires_input_gradient(criterion):
    #     inputs.requires_grad = True

    with (
        contextlib.nullcontext()
            if is_train
                else torch.no_grad()
    ):

        outputs = model(inputs)

        if unlabeled_batch is None:
            unlabeled_outputs = None
            unlabeled_targets = None
        else:
            need_to_freeze = hasattr(model, "keep_active_indices")
            if need_to_freeze:
                model.keep_active_indices = True
            # labels are not dropped yet
            unlabeled_targets = unlabeled_batch[1].to(inputs.device)
            unlabeled_batch = unlabeled_batch[0].to(inputs.device)
            unlabeled_outputs = model(unlabeled_batch)
            if need_to_freeze:
                model.keep_active_indices = False

        metrics = make_metrics_dict(
            compute_metrics,
            targets,
            outputs,
            is_train
        )

    toggle_logging = log_this_batch and hasattr(criterion, "log_this_batch")
    if toggle_logging:
        criterion.log_this_batch = True

    if is_train:

        # if not isinstance(criterion, DiverseGradientsLoss):
        #     outputs = normalize_outputs(outputs)
        # outputs = normalize_outputs(outputs)

        train_info = do_default_train(
            model,
            freeze_model,
            criterion,
            optimizer,
            outputs,
            targets,
            logger,
            unlabeled_outputs=unlabeled_outputs,
            log_this_batch=log_this_batch,
            unlabeled_targets=unlabeled_targets
        )

        if (
                hasattr(model, "single_model_id")
            and
                model.single_model_id is not None
        ):
            suffix = f"_{model.single_model_id}"
            train_info_for_stage = train_info[TRAIN_LOGS_NAME]
            train_info[TRAIN_LOGS_NAME] = overwrite_keys(
                train_info_for_stage,
                suffix
            )
            metrics = overwrite_keys(metrics, suffix)

    else:

        if isinstance(targets, dict) or criterion is None:
            # don't compute loss when have multiple labels or no criterion
            train_info = None
        else:
            loss_for_backward, loss_info, gradients_info = compute_criterion(
                outputs,
                targets,
                criterion
            )
            assert loss_for_backward.requires_grad == False
            train_info = {stage_name: loss_info | gradients_info}

    if toggle_logging:
        criterion.log_this_batch = False

    return (
        metrics,
        outputs,
        train_info
    )


# def compute_diversity_measure(ground_truth_probs):
#     assert len(ground_truth_probs) > 0
#     result = 0
#     denominator = 0

#     for probs in ground_truth_probs.values():
#         probs = np.mean(np.array(probs))
#         result += probs * probs
#         denominator += probs

#     if denominator == 0:
#         result = 0
#     else:
#         result /= denominator

#     return result


def make_metrics_dict(
    compute_metrics,
    targets,
    outputs,
    is_train
):

    def get_metrics(targets, outputs, compute_metrics):

        if len(targets.shape) > 1 and targets.shape[1] == 1:
            targets = torch.squeeze(targets, dim=1)

        return compute_metrics(outputs, targets)

    metrics = {}

    if isinstance(targets, dict):
        assert not is_train

        for target_key, targets_value in targets.items():
            metrics[
                make_name_from_prefix_and_key(
                    compute_metrics.metrics_base_name,
                    target_key
                )
            ] = get_metrics(
                targets_value,
                outputs,
                compute_metrics
            )
    else:

        metrics[compute_metrics.metrics_base_name] = get_metrics(
            targets,
            outputs,
            compute_metrics
        )

    return metrics


# def is_wandb_only_stat(nested_key_as_list):

#     stat_name_pos = -1

#     if BASE_ESTIMATOR_LOG_SUFFIX in nested_key_as_list[stat_name_pos]:
#         assert len(nested_key_as_list) > 1
#         stat_name_pos -= 1

#     if nested_key_as_list[stat_name_pos].split(NAME_SEP)[0] \
#         in WANDB_ONLY_STATS:

#         return True

#     return False


def assert_wandb(wandb_run, batchwise_statistics, statistics_config):
    assert wandb_run
    wandb_stat_config = statistics_config["wandb"]["stats"]
    if not batchwise_statistics:
        for wandb_stat in WANDB_ONLY_STATS:
            if wandb_stat == STAT_WANDB_INPUT:
                continue
            if (wandb_stat_config[wandb_stat]):
                raise Exception(
                    "Can't log {} in wandb "
                    "for epoch-wise statistics".format(wandb_stat)
                )


def log_stats_in_wandb(
    logger,
    stage_name,
    stats_to_log,
    media_stats,
    metrics_names,
    losses_names,
    to_log_metrics,
    to_log_losses,
    step
):

    def extract_stats_by_names(all_stats, names_to_extract):
        return {
            name_to_extract: all_stats.pop(name_to_extract)
                for name_to_extract in names_to_extract
        }

    def merge_stats_back(all_stats, extracted_stats):
        for extracted_stat_name, extracted_stat in extracted_stats.items():
            all_stats[extracted_stat_name] = extracted_stat

    metrics_stats = {}
    losses_stats = {}

    stats_to_log_for_current_stage = stats_to_log[stage_name]
    media_stats_for_current_stage = media_stats[stage_name]

    if not to_log_metrics:
        metrics_stats = extract_stats_by_names(
            stats_to_log_for_current_stage,
            metrics_names
        )

    if not to_log_losses:
        losses_stats = extract_stats_by_names(
            stats_to_log_for_current_stage,
            losses_names
        )

    assert dicts_with_non_intersecting_keys(
        stats_to_log_for_current_stage,
        media_stats_for_current_stage
    )

    try_to_log_in_wandb(
        logger,
        {
            stage_name: (
                stats_to_log_for_current_stage | media_stats_for_current_stage
            )
        },
        step=step
    )

    merge_stats_back(stats_to_log_for_current_stage, metrics_stats)
    merge_stats_back(stats_to_log_for_current_stage, losses_stats)


def do_default_train(
    model,
    freeze_model,
    criterion,
    optimizer,
    outputs,
    targets,
    logger=None,
    unlabeled_outputs=None,
    log_this_batch=False,
    unlabeled_targets=None
):
    loss_for_backward, loss_info, gradients_info = compute_criterion(
        outputs,
        targets,
        criterion,
        unlabeled_outputs=unlabeled_outputs,
        unlabeled_targets=unlabeled_targets
    )

    train_info = {TRAIN_LOGS_NAME: loss_info | gradients_info}

    optimizer.zero_grad(set_to_none=True)
    loss_for_backward.backward()
    if not freeze_model:
        optimizer.step()

    return train_info


def compute_criterion(
    outputs,
    targets,
    criterion,
    unlabeled_outputs=None,
    unlabeled_targets=None
):

    if isinstance(criterion, tuple):
        assert len(criterion) == 2
        unlabeled_criterion = criterion[1]
        criterion = criterion[0]
        if hasattr(unlabeled_criterion, "smoothing_eps"):
            assert getattr(unlabeled_criterion, "smoothing_eps") is None, \
                "only main criterion can have smoothing_eps"
    else:
        unlabeled_criterion = None

    if hasattr(criterion, "smoothing_eps"):
        smoothing_eps = criterion.smoothing_eps
    else:
        smoothing_eps = None

    onehot_targets = ensure_targets_shape(
        outputs,
        targets,
        smoothing_eps
    ).float()

    unlabeled_criterion_output = None

    if unlabeled_outputs is None:

        criterion_output = criterion(outputs, onehot_targets)

    else:
        if unlabeled_criterion is not None:
            unlabeled_onehot_targets = ensure_targets_shape(
                unlabeled_outputs,
                unlabeled_targets
            ).float()
            criterion_output = criterion(outputs, onehot_targets)

            # unlabeled criterion takes unlabeled_outputs
            # similarly to how criterion takes ordinary outputs
            unlabeled_criterion_output = unlabeled_criterion(
                unlabeled_outputs,
                unlabeled_onehot_targets
            )
        else:
            criterion_output = criterion(outputs, onehot_targets, unlabeled_outputs)

    loss_for_backward, loss_info, gradients_info = extract_loss(
        criterion_output
    )

    if unlabeled_criterion_output is not None:
        (
            loss_for_backward_from_unlabeled,
            loss_info_from_unlabeled,
            gradients_info_unlabeled
        ) \
            = extract_loss(
                unlabeled_criterion_output
        )
        assert len(gradients_info_unlabeled) == 0
        assert len(gradients_info) == 0
        loss_info_from_unlabeled = {
            "unlabeled_" + k: v for k, v in loss_info_from_unlabeled.items()
        }
        loss_for_backward += loss_for_backward_from_unlabeled
        loss_info |= loss_info_from_unlabeled

    return loss_for_backward, loss_info, gradients_info


def extract_loss(criterion_output):
    if isinstance(criterion_output, tuple):
        assert len(criterion_output) == 3
        loss_for_backward = criterion_output[0]
        loss_info = criterion_output[1]
        assert isinstance(loss_info, dict)
        gradients_info = criterion_output[2]
        # if len(gradients_info) > 0:
        #     gradients_info = {INPUT_GRAD_NAME: gradients_info}
    else:
        loss_for_backward = criterion_output
        loss_info = {LOSS_STATISTICS_NAME: loss_for_backward.item()}
        gradients_info = {}
    return loss_for_backward, loss_info, gradients_info


# def check_optimizer(optimizer, expected_optimizer):
#     if expected_optimizer == "sgd":
#         assert isinstance(optimizer, optim.SGD), "Expected optim.SGD"
#     elif expected_optimizer == "adam":
#         assert isinstance(optimizer, optim.Adam), "Expected optim.Adam"
#     else:
#         raise_unknown(
#             "expected_optimizer",
#             expected_optimizer,
#             "checkpoint"
#         )


# def check_lr_scheduler(lr_scheduler, expected_lr_scheduler):
#     if expected_lr_scheduler == "reduce_on_plateau":
#         assert isinstance(
#             lr_scheduler,
#             torch.optim.lr_scheduler.ReduceLROnPlateau
#         ), "Expected torch.optim.lr_scheduler.ReduceLROnPlateau"
#     elif expected_lr_scheduler == "constant":
#         assert isinstance(
#             lr_scheduler,
#             torch.optim.lr_scheduler.ConstantLR
#         ), "Expected torch.optim.lr_scheduler.ConstantLR"
#     elif expected_lr_scheduler.startswith(FROM_CLASS_KEY):
#         pass
#     else:
#         raise_unknown(
#             "expected_lr_scheduler",
#             expected_lr_scheduler,
#             "checkpoint"
#         )


def make_or_get_from_checkpoint(
    name,
    checkpoint,
    config,
    check_func,
    make_func,
    **make_args
):
    object_config = config[name]
    checkpoint_content = checkpoint.get(name)
    if checkpoint_content is not None:
        if check_func is not None:
            check_func(checkpoint_content, object_config["type"])
        result = checkpoint_content
    else:
        result = make_func(object_config, **make_args)
        checkpoint[name] = result
    return result


def ensure_targets_shape(outputs, targets, smoothing_eps=None):
    # outputs = normalize_outputs(outputs)
    if isinstance(outputs, list):
        assert outputs
        assert len(outputs[0]) == 2
        outputs = outputs[0][1]
    if outputs.shape == targets.shape:
        pass
    elif outputs.shape[:-1] == targets.shape:
        last_dim = outputs.shape[-1]
        if last_dim == 1:
            targets = targets.unsqueeze(-1)
        else:
            num_classes = last_dim
            targets = torch.nn.functional.one_hot(targets, num_classes)
    else:
        raise Exception(
            "Could not make same shape for outputs"
            " with shape {} and targets with shape {}".format(
                outputs.shape,
                targets.shape
            )
        )

    # TODO(Alex | 26.07.2024): add smoothing directly to criterion
    if smoothing_eps is not None:
        targets = smooth_labels(targets, smoothing_eps)

    return targets


# def make_base_estimator_name(base_estimator_id):
#     return "{} {}".format(
#         BASE_ESTIMATOR_LOG_SUFFIX,
#         base_estimator_id
#     )


# TODO(Alex | 25.07.2024): simplify or even remove this function
def make_batch_stats(
    experiment_config: Dict[str, Any],
    metrics: Dict[str, Union[torch.tensor, List[torch.tensor]]],
    inputs: torch.tensor,
    outputs: torch.tensor,
    targets: Union[torch.tensor, Dict[str, torch.tensor]],
    train_info: Dict[str, Dict[str, float]],
    batch_idx: int,
    epoch: int,
    stage_name: str,
    idx: int = 0
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    List[str],
    List[str]
]:
    """
    Prepares current stage's train/eval statistics for logging.
    Where stage means "run_epoch" call on a certain dataloader.

    Always prepares metrics and losses.

    If <experiment_config>["use_wandb"] is True then depending on the values
    in <experiment_config>["wandb"]["stats"] can prepare for logging in wandb:

        - inputs and optionally input gradients if they exist
            (input gradients can't be logged without logging inputs).

        - logits
            (there is an issue: https://github.com/wandb/wandb/issues/1206).

        - prediction.

        - target.

    Args:

        experiment_config (Dict[str, Any]): cameo.

        metrics (
            Dict[
                str,
                Union[torch.tensor as scalar, List[torch.tensor as scalar]]
            ]
        ):
            a dictionary that maps metrics name to the metrics
            for the current batch.
            Metrics name contains the name of the function used for the metrics
            computation (e.g. "Accuracy"). In case of dataloader
            with multiple labels, it is also concatenated with label names.
            Metrics for the current batch
            is either a scalar for a single model
            or a list of scalars for the ensemble model.

        inputs (torch.tensor of shape [B, C, H, W]): model inputs
            to log in wandb if requested in config. It will be logged
            as a picture with caption.
            The caption will contain information about ground truth label
            and model's prediction.
            Where B - batchsize, C - number of channels in an image,
            H - height of an image, W - width of an image.

        outputs (torch.tensor of shape [B, Y]): model outputs
            from which predictions are extracted to log in wandb
            if requested in config.
            Where B as for inputs, Y - number of classes.

        targets (
            Union[
                torch.tensor of shape [B],
                Dict[str, torch.tensor of shape [B]]
            ]
        ): in case of a dataloader with multiple labels,
            it is a dictionary that maps label name
            to the batch of its corresponding ground truth values.
            For a dataloader with single label it is a batch
            of ground truth values.
            Where B as for inputs.

        train_info (Dict[str, Dict[str, float]]): dictionary
            that maps stage name to the loss dictionary
            and optionally gradients dictionary if it exists.
            These dictionaries are returned by the "do_train_func"
            in the "run_epoch".
            Loss dictionary maps loss names to their corresponding values.
            Gradients dictionary maps "INPUT_GRAD_NAME" to the input gradient.

        batch_idx (int): index of the current batch,
            needed for the input caption.

        epoch (int): index of the current epoch, needed for the input caption.

        stage_name (str): name of the current stage
            (stage is defined in this function's description).

        idx (int): which batch element to extract for logging.
            Default: 0

    Returns:

        tuple of the following objects:

            batch_stats (Dict[str, Dict[str, Any]]): same as <train_info>
                but inner dictionaries might also contain info about wandb
                logged elements
                (except for those that are stored in batch_media):
                    - metrics.
                    - logits.
                    - prediction.
                    - targets.

            batch_media (Dict[str, Dict[str, Any]]): same as <train_info>
                but inner dictionaries might contain info about inputs
                and input gradients if they are logged.

            metric_names (List[str]): list of names for logged metrics;
                computed as list(<metrics>.keys()).

            losses_names (List[str]): list of names for logged losses
                from the <train_info>.
    """

    def extract_values(
        raw_values,
        extractor=lambda x: x,
        func=lambda x: x.item(),
        aggregate_if_list=True
    ):
        # ensemble case
        if isinstance(raw_values, list):
            assert raw_values
            if isinstance(raw_values[0], list):
                assert len(raw_values[0]) == 2
            else:
                assert torch.is_tensor(raw_values[0])
            result = {
                make_base_estimator_name(i):
                    func(extractor(raw_values[i]))
                        for i in range(len(raw_values))
            }

            if aggregate_if_list:
                values_array = np.array(list(result.values()))
                add_aggregated_stats(result, values_array)

            return result

        else:
            # single model case
            return func(raw_values)

    def make_ground_truth_probs_dict(outputs, target, func):
        assert len(outputs)
        assert len(outputs[0]) == 2
        onehot_target = ensure_targets_shape(
            outputs,
            target
        )
        return {
            i: (
                (
                    func(outputs_per_model[1].detach()) * onehot_target
                ).sum().cpu().item() / onehot_target.shape[0]
            )
                for i, outputs_per_model in enumerate(outputs)
        }

    # def prepare_input_grad_for_image(
    #     input_grad,
    #     inputs_shape,
    #     idx
    # ):
    #     input_grad_reshaped = torch.abs(
    #         input_grad.detach().view(inputs_shape)[idx].squeeze()
    #     )
    #     input_grad_reshaped = (
    #         input_grad_reshaped / input_grad_reshaped.max()
    #     )
    #     return input_grad_reshaped

    # outputs = normalize_outputs(outputs)

    batch_media = {stage_name: {}}
    batch_media_for_current_stage = batch_media[stage_name]
    if train_info is not None:

        train_info_for_current_stage = train_info[stage_name]
        # if INPUT_GRAD_NAME in train_info_for_current_stage:

        #     batch_media_for_current_stage[INPUT_GRAD_NAME] \
        #         = train_info_for_current_stage.pop(INPUT_GRAD_NAME)
        losses_names = [
            stat_name for stat_name
                in train_info_for_current_stage.keys()
                    if LOSS_STATISTICS_NAME in stat_name
        ]
        batch_stats = train_info

    else:
        batch_stats = {stage_name: {}}
        losses_names = []

    batch_stats_for_current_stage = batch_stats[stage_name]
    assert isinstance(metrics, dict)
    for metrics_key, metrics_value in metrics.items():

        if isinstance(outputs, list):
            assert isinstance(metrics_value, list)

        batch_stats_for_current_stage[metrics_key] = extract_values(
            metrics_value
        )

    statistics_config = experiment_config["statistics"]

    # # multi label dataloader for ensemble
    # if isinstance(outputs, list) and isinstance(targets, dict):
    #     softmax = torch.nn.Softmax(dim=-1)
    #     batch_stats_for_current_stage[GROUND_TRUTH_PROBS_STAT_KEY] = {}
    #     for target_name, target in targets.items():
    #         batch_stats_for_current_stage[GROUND_TRUTH_PROBS_STAT_KEY][
    #             target_name
    #         ] = make_ground_truth_probs_dict(outputs, target, softmax)

    if statistics_config["use_wandb"]:

        prediction = None
        target_stats = None
        input_grad = None

        wandb_config = statistics_config["wandb"]

        wandb_stats_config = wandb_config["stats"]

        to_store_input = wandb_stats_config[STAT_WANDB_INPUT]

        if wandb_stats_config[STAT_WANDB_LOGITS]:

            batch_stats_for_current_stage[STAT_WANDB_LOGITS] = extract_values(
                outputs,
                extractor=lambda x: x[1],
                func=(lambda x: x[idx].detach().squeeze().cpu().numpy()),
                aggregate_if_list=False
            )

        if (
            wandb_stats_config[STAT_WANDB_PREDICTION]
                or to_store_input
        ):

            prediction = extract_values(
                outputs,
                extractor=lambda x: x[1],
                func=lambda x: torch.argmax(x[idx]).item(),
                aggregate_if_list=False
            )

            if wandb_stats_config[STAT_WANDB_PREDICTION]:
                batch_stats_for_current_stage[STAT_WANDB_PREDICTION] \
                    = prediction

        if (
            wandb_stats_config[STAT_WANDB_TARGET]
                or to_store_input
        ):

            target_stats = {}

            if isinstance(targets, dict):
                targets_names = []
                for target_key, target_value in targets.items():
                    targets_names.append(
                        make_name_from_prefix_and_key("target", target_key)
                    )

                    target_stats[targets_names[-1]] = extract_values(
                        target_value,
                        func=lambda x: x[idx].item()
                    )

            else:

                target_stats["target"] = extract_values(
                    targets,
                    func=lambda x: x[idx].item()
                )

            if wandb_stats_config[STAT_WANDB_TARGET]:
                batch_stats_for_current_stage |= target_stats

        if to_store_input:

            assert prediction is not None
            assert target_stats

            caption = make_caption(
                idx,
                batch_idx,
                epoch + 1,
                target_stats=(target_stats | {"prediction" : prediction})
            )

            input_image = inputs[idx].detach().cpu().squeeze()

            wandb_input_image = wandb.Image(
                input_image,
                caption=caption
            )

            batch_media_for_current_stage[STAT_WANDB_INPUT] = wandb_input_image

            # TODO(Alex | 11.09.2023): uncomment when gradients will be fixed
            # to work with feature_extractor for ensemble

            # if INPUT_GRAD_NAME in batch_media_for_current_stage:

            #     gradients_info = batch_media_for_current_stage[INPUT_GRAD_NAME]
            #     assert isinstance(gradients_info, dict)
            #     for base_estimator_name, input_grad in gradients_info.items():

            #         input_grad_reshaped = prepare_input_grad_for_image(
            #             input_grad,
            #             inputs.shape,
            #             idx
            #         )

            #         input_and_input_grad = torch.cat(
            #             [input_image, input_grad_reshaped],
            #             dim=-1
            #         )
            #         gradient_caption = make_caption(
            #             idx,
            #             batch_idx,
            #             epoch + 1,
            #             estimator_name=base_estimator_name,
            #             and_grad=True
            #         )
            #         gradients_info[
            #             base_estimator_name
            #         ] = wandb.Image(
            #             input_and_input_grad,
            #             caption=gradient_caption
            #         )

        # else:

        #     if INPUT_GRAD_NAME in batch_media_for_current_stage:
        #         batch_media_for_current_stage.pop(INPUT_GRAD_NAME)

    return (
        batch_stats,
        batch_media,
        list(metrics.keys()),
        losses_names
    )


def add_aggregated_stats(stats, values_array):
    stats[MEAN_STAT_NAME] = np.mean(values_array)
    stats[MAX_STAT_NAME] = np.max(values_array)
    stats[MIN_STAT_NAME] = np.min(values_array)
    stats[STD_STAT_NAME] = np.std(values_array)


def make_caption(
    input_idx,
    batch_idx,
    epoch,
    estimator_name=None,
    and_grad=False,
    target_stats=None
):
    result = "Input number {} in batch {} for epoch {}".format(
        input_idx,
        batch_idx,
        epoch
    )
    if and_grad:
        assert target_stats is None
        result += " and it's gradient"
    if estimator_name is not None:
        result += " for {}".format(estimator_name)
    if target_stats is not None:
        assert not and_grad
        if PREDICTION_STAT_KEY in target_stats:
            result += "\n{}: {}".format(
                PREDICTION_STAT_KEY,
                target_stats[PREDICTION_STAT_KEY]
            )
        for target_name, target_value in sorted(target_stats.items()):
            if target_name == PREDICTION_STAT_KEY:
                continue
            result += "\n{}: {}".format(target_name, target_value)

    return result


def make_name_from_prefix_and_key(prefix, key):
    return prefix + NAME_SEP + str(key)


def aggregate_stats(stats, only_mean=False):
    aggregated_stats = {}
    recompute_modelwise_aggregated_stats = False

    for key, stat_history in stats.items():

        if key in AGGREGATED_STAT_NAMES:
            recompute_modelwise_aggregated_stats = True
            continue

        if isinstance(stat_history, dict):
            aggregated_stats[key] = aggregate_stats(
                stat_history,
                only_mean=only_mean
            )
        else:
            assert isinstance(stat_history, list)
            stat_history_array = np.array(stat_history)
            aggregated_stats[key] = {}
            aggregated_stats_for_key = aggregated_stats[key]
            aggregated_stats_for_key["mean"] = np.mean(stat_history_array)
            if only_mean:
                aggregated_stats[key] = aggregated_stats_for_key["mean"]
            else:
                aggregated_stats_for_key["first"] = stat_history_array[0]
                aggregated_stats_for_key["last"] = stat_history_array[-1]
                aggregated_stats_for_key["std"] = np.std(stat_history_array)
                aggregated_stats_for_key["max"] = np.max(stat_history_array)
                aggregated_stats_for_key["min"] = np.min(stat_history_array)

    if recompute_modelwise_aggregated_stats:
        values = []
        # iterate over values for different base_estimators
        for aggregated_value in aggregated_stats.values():
            if isinstance(aggregated_value, dict):
                assert "mean" in aggregated_value
                values.append(aggregated_value["mean"])
            else:
                values.append(aggregated_value)
        values_array = np.array(values)
        add_aggregated_stats(aggregated_stats, values_array)

    return aggregated_stats


# TODO(Alex | 25.07.2024): simplify or even remove this function
def log_stats(
    logger: RedneckLogger,
    stats: Dict[
        str,
        Dict[
            str,
            Union[
                float,
                Dict[str, float]
            ]
        ]
    ],
    stage_name: str,
    epoch: int
):
    """
    Log flattened stats into logs stdout and stderr
    as well as in csv file if it exists.

    For each stage (<stage_name> is the same as in "make_batch_stats")
    log pairs of stat name and its corresponding stat value.
    Stat names are made by chaining keys from the nested <stats> dict
    with "NESTED_LOGS_SEPARATOR" as separator.
    Apply <make_csv_column_name> to the names of stats to define column names
    for logging corresponding stat values in a csv.
    """

    def get_name_and_value_pairs_to_log(name_and_value_pairs, stats, prefix=""):

        only_mean = True
        for stat_name, stat_value in stats.items():

            value_name = prefix + str(stat_name)
            value_as_str = None
            # aggregated leaf stats
            if isinstance(stat_value, dict) and (
                "mean" in stat_value
                    and "std" in stat_value
                    and "first" in stat_value
                    and "last" in stat_value
                    and "max" in stat_value
                    and "min" in stat_value
            ):
                only_mean = False
                value_as_str = (
                    "{:0.4f}"
                    "{}"
                    "{:0.4f} | "
                    "{:0.4f} | "
                    "{:0.4f} | "
                    "{:0.4f} | "
                    "{:0.4f}".format(
                        stat_value["mean"],
                        AFTER_MEAN_SEP,
                        stat_value["std"],
                        stat_value["first"],
                        stat_value["last"],
                        stat_value["max"],
                        stat_value["min"]
                    )
                )
            # intermediate stats
            elif isinstance(stat_value, dict):

                name_and_value_pairs, only_mean \
                    = get_name_and_value_pairs_to_log(
                        name_and_value_pairs,
                        stat_value,
                        prefix="{}{}{}".format(
                            prefix,
                            stat_name,
                            NESTED_LOGS_SEPARATOR
                        )
                    )
            else:

                # if stat_value > 1e-4:
                #     value_as_str = "{:0.4f}".format(stat_value)
                # else:
                #     value_as_str = "{:2E}".format(stat_value)
                value_as_str = value_as_str_scientific(stat_value)

            if value_as_str is not None:
                name_and_value_pairs.append((value_name, value_as_str))

        # to mark end of pairs from current stats
        name_and_value_pairs.append((None, 0))

        return name_and_value_pairs, only_mean

    logger.log_separator()

    name_and_value_pairs_to_log, only_mean = get_name_and_value_pairs_to_log(
        [],
        stats[stage_name],
        prefix=""
    )

    name_and_value_pairs_to_log = remove_elements_from_the_end(
        name_and_value_pairs_to_log,
        (None, 0)
    )

    flattened_stats = {}

    log_msg = ""
    for stat_name, value in name_and_value_pairs_to_log:
        # stats separators
        if stat_name is None:
            if value == 0:
                log_msg += "\n"
            else:
                raise_unknown(
                    "Separator code",
                    value,
                    "log_stats"
                )
        else:

            flattened_stats[stat_name] = value

            log_msg += "{}{}:\n {}{} |{}|\n".format(
                INDENT,
                stat_name,
                INDENT,
                INDENT,
                value
            )

    if only_mean:
        statistics_description = "| mean |"
    else:
        statistics_description = "| mean +- std | first | last | max | min |"

    log_msg = (
        "\nEpoch {} stats for \"{}\" ({}):\n\n").format(
        epoch + 1,
        stage_name,
        statistics_description
    ) + log_msg
    logger.log(log_msg)
    logger.log_separator()

    return flattened_stats


def value_as_str_scientific(value):
    if value > SCIENTIFIC_NOTATION_THRESHOLD:
        value_as_str = "{:0.4f}".format(value)
    else:
        value_as_str = "{:2E}".format(value)
    return value_as_str


def log_stats_in_csv(
    all_stats,
    keep_modelwise,
    to_log_in_csv,
    make_csv_column_name,
    logger
):

    def keep_in_csv(stat_name, keep_modelwise):

        if keep_modelwise:
            for key in AGGREGATED_STAT_NAMES:
                if NESTED_LOGS_SEPARATOR + key in stat_name:
                    return False
        else:
            return not BASE_ESTIMATOR_LOG_SUFFIX in stat_name

        return True

    def whether_to_log_in_csv(
        stat_name,
        stage_name,
        keep_modelwise
    ):
        return (
                keep_in_csv(stat_name, keep_modelwise)
            and
                (
                        not TRAIN_LOGS_NAME in stage_name
                    or
                        LOSS_STATISTICS_NAME in stat_name
                )
        )

    if (
            to_log_in_csv
        and
            logger.csv_output is not None
    ):

        for stage_name, stats in all_stats.items():

            assert isinstance(stats, dict)

            for stat_name, value in stats.items():

                if whether_to_log_in_csv(
                    stat_name,
                    stage_name,
                    keep_modelwise
                ):
                    # log only mean value in csv for batchwise case
                    if AFTER_MEAN_SEP in value:
                        value_for_csv = value.split(AFTER_MEAN_SEP)[0]
                    else:
                        value_for_csv = value
                    try_to_log_in_csv(
                        logger,
                        make_csv_column_name(stage_name, stat_name),
                        value_for_csv
                    )


def make_csv_column_name_default(stage_name, stat_name):

    return "\"{}\" {}".format(
        stage_name,
        stat_name
    )


# def get_csv_column_name_maker(dataset_name):

#     def make_csv_column_name_default(stage_name, stat_name):

#         return "\"{}\" {}".format(
#             stage_name,
#             stat_name
#         )

#     def make_csv_column_name_for_features_labeller(stage_name, stat_name):

#         stat_name = stat_name.replace(
#             BASE_ESTIMATOR_LOG_SUFFIX,
#             SHORT_BASE_ESTIMATOR_LOG_SUFFIX
#         )
#         csv_stat_name = stat_name

#         if (
#             stage_name == EVAL_ON_TRAIN_LOGS_NAME
#                 or stage_name == TRAIN_LOGS_NAME
#         ):
#             csv_stage_name = "diag train"
#         elif (
#             DIAG_COMMON_PREFIX in stage_name
#                 and not OFF_DIAG_COMMON_PREFIX in stage_name
#         ):
#             csv_stage_name = "diag eval"
#         elif OFF_DIAG_COMMON_PREFIX in stage_name:

#             # diversity measure stat should conatain
#             # only "NAME_SEP" name separator
#             if DIVERSITY_MEASURE_STAT_KEY in stat_name:
#                 assert NESTED_LOGS_SEPARATOR in stat_name
#                 stat_name = stat_name.replace(
#                     NESTED_LOGS_SEPARATOR,
#                     NAME_SEP,
#                     1
#                 )

#             # multi label case
#             if OFF_DIAG_COMMON_PREFIX in stat_name:
#                 assert NAME_SEP in stat_name
#                 stat_name_split = stat_name.split(NAME_SEP)
#                 assert len(stat_name_split) > 2
#                 csv_stat_name = stat_name_split[0]
#                 assert stat_name_split[1] == OFF_DIAG_COMMON_PREFIX
#                 # ensemble case
#                 if NESTED_LOGS_SEPARATOR in stat_name_split[2]:
#                     stat_name_subsplit = stat_name_split[2].split(
#                         NESTED_LOGS_SEPARATOR
#                     )
#                     csv_stage_name = stat_name_subsplit[0]
#                     csv_stat_name += NESTED_LOGS_SEPARATOR \
#                         + stat_name_subsplit[1]
#                 # single model case
#                 else:
#                     csv_stage_name = stat_name_split[2]
#             # single label case
#             else:
#                 csv_stage_name = stage_name

#         else:

#             csv_stage_name = stage_name

#         if NESTED_LOGS_SEPARATOR in csv_stat_name:
#             stat_name_split = csv_stat_name.split(NESTED_LOGS_SEPARATOR)
#             assert len(stat_name_split) == 2
#             csv_stat_name = "{} {}".format(
#                 stat_name_split[1],
#                 stat_name_split[0]
#             )

#         return "{} ({})".format(csv_stage_name, csv_stat_name).replace(
#             SHORT_BASE_ESTIMATOR_LOG_SUFFIX,
#             BASE_ESTIMATOR_LOG_SUFFIX
#         )


#     if dataset_name == "features_labeller":
#         return make_csv_column_name_for_features_labeller
#     else:
#         return make_csv_column_name_default


def make_scheduler_step(lr_scheduler, epoch_stats_for_current_stage):

    if lr_scheduler is None:
        return

    if lr_scheduler.warmup is None:
        warmup_context = contextlib.nullcontext()
    else:
        warmup_context = lr_scheduler.warmup.dampening()

    with warmup_context:
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            assert hasattr(lr_scheduler, "loss_stat_name")
            loss_stat_name = lr_scheduler.loss_stat_name
            if loss_stat_name is None:
                loss_stat_name = LOSS_STATISTICS_NAME

            loss_stat = get_with_assert(
                epoch_stats_for_current_stage,
                loss_stat_name
            )
            lr_scheduler.step(
                loss_stat
            )
        else:
            lr_scheduler.step()


def init_empty_checkpoint(experiment_name):
    checkpoint = {key: value for key, value in CHECKPOINT_TEMPLATE}
    checkpoint[EXP_NAME_KEY] = experiment_name
    return checkpoint


def init_checkpoint(
    starting_checkpoint_path=None,
    check_only_model=False,
    experiment_name=None,
    logger=None,
    load_only_model=False
):
    """
    checkpoint contents:
      CHECKPOINT_TEMPLATE
    """

    loaded_checkpoint = {}
    empty_checkpoint = {}

    if starting_checkpoint_path is not None:
        log_or_print(
            "Reading checkpoing from {}..".format(starting_checkpoint_path),
            logger=logger,
            auto_newline=True
        )
        loaded_checkpoint = read_checkpoint(starting_checkpoint_path)

        if load_only_model:
            logger.log("Taking only model from checkpoint.")
            experiment_name = loaded_checkpoint[EXP_NAME_KEY]
            loaded_checkpoint = {MODEL_KEY: loaded_checkpoint[MODEL_KEY]}

    if len(loaded_checkpoint) == 0 or load_only_model:
        assert experiment_name is not None
        empty_checkpoint = init_empty_checkpoint(
            experiment_name
        )

    checkpoint = empty_checkpoint | loaded_checkpoint

    return checkpoint


def get_checkpoint(experiment_config, logger=None):
    checkpoint_config = experiment_config.get("checkpoint")
    if (
        checkpoint_config is None
    ):

        checkpoint = init_checkpoint(
            experiment_name=experiment_config[EXP_NAME_KEY],
            logger=logger
        )

    else:
        checkpoint_path = get_with_assert(
            checkpoint_config,
            "starting_checkpoint_path"
        )
        checkpoint = init_checkpoint(
            starting_checkpoint_path
                =checkpoint_path,
            check_only_model=checkpoint_config.get("check_only_model", True),
            load_only_model=checkpoint_config.get("load_only_model", False),
            logger=logger
        )

    return checkpoint


# def normalize_outputs(outputs):
#     if is_diverse_vit_output(outputs):
#         outputs = outputs[0]
#     return outputs


def make_infinite_data_iterator(dataloader):
    return InfiniteDataIterator(dataloader)


# taken from: https://github.com/p-lambda/wilds/blob/472677590de351857197a9bf24958838c39c272b/examples/utils.py#L385
class InfiniteDataIterator:
    """
    Adapted from https://github.com/thuml/Transfer-Learning-Library

    A data iterator that will never stop producing data
    """

    def __init__(self, data_loader: torch.utils.data.DataLoader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            print("Reached the end, resetting data loader...")
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


# taken from: https://github.com/yoonholee/DivDis/blob/b9de1a637949594054240254f667063788ee1573/wilds/utils.py#L387
def collate_list(vec):
    """
    If vec is a list of Tensors, it concatenates them all along the first dimension.

    If vec is a list of lists, it joins these lists together, but does not attempt to
    recursively collate. This allows each element of the list to be, e.g., its own dict.

    If vec is a list of dicts (with the same keys in each dict), it returns a single dict
    with the same keys. For each key, it recursively collates all entries in the list.
    """
    if not isinstance(vec, list):
        raise TypeError("collate_list must take in a list")
    elem = vec[0]
    if torch.is_tensor(elem):
        return torch.cat(vec)
    elif isinstance(elem, list):
        return [obj for sublist in vec for obj in sublist]
    elif isinstance(elem, dict):
        return {k: collate_list([d[k] for d in vec]) for k in elem}
    else:
        raise TypeError("Elements of the list to collate must be tensors or dicts.")
