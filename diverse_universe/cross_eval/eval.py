import os
import sys
import torch
import shutil
from sklearn.metrics import roc_auc_score
import numpy as np
from tqdm import tqdm
from stuned.utility.utils import (
    raise_unknown,
    get_with_assert,
    pretty_json,
    log_or_print,
    append_dict,
)
from stuned.utility.logger import (
    make_logger,
    try_to_log_in_csv
)
from stuned.utility.configs import (
    RUN_PATH_CONFIG_KEY
)


# local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from diverse_universe.train.losses import (
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
)
from diverse_universe.local_models.ensemble import (
    is_ensemble,
    get_ensemble,
    make_ensembles
)
from diverse_universe.train.losses import (
    get_metrics_mapping
)
from diverse_universe.local_models.common import (
    wrap_model,
    make_models_dict_from_huge_string
)
from diverse_universe.local_datasets.common import (
    make_dataloaders,
    make_cached_dataloaders
)
from diverse_universe.local_datasets.wilds import (
    metadata_to_group
)
sys.path.pop(0)


# TODO(Alex | 11.05.2024): move to utils
ADJUSTED_GROUPWISE_KEY = "_adjusted"

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
    ]

    recompute_all = experiment_config.get("recompute_all", False)

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
            metrics_mappings=metrics_mappings,
            prune_metrics=[],
            evaluation_kwargs=eval_kwargs,
            logger=logger,
            model_to_prop_dict=model_to_prop_dict,
            results_key=eval_type,
            recompute_all=recompute_all
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
                "metrics_mappings": metrics_mappings,
                "cache": {},
                "id_dataloader": id_dataloader # TODO(Alex | 06.05.2024): specify by config
            },
            model_to_prop_dict=model_to_prop_dict,
            logger=logger,
            results_key=eval_type,
            recompute_all=recompute_all
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
    elif exp_type == "debug":
        return prepare_debug_dataloaders(cached_datasets_info)
    else:
        raise_unknown(exp_type, "experiment_type", "prepare_dataloaders")


def prepare_debug_dataloaders(cached_datasets_info):
    deit3b_2layers_all_dataloaders_dict = make_cached_dataloaders(
        subset_dict_by_keys(
            get_with_assert(cached_datasets_info, "path_mapping"),
            ["in_val", "imagenet_a", "imagenet_r" ]
        )
    )
    deit3b_2layers_all_dataloaders_dict = {
        "imagenet_a": (deit3b_2layers_all_dataloaders_dict["imagenet_a"], "ina"),
        "imagenet_r": (deit3b_2layers_all_dataloaders_dict["imagenet_r"], "inr"),
        "in_val": deit3b_2layers_all_dataloaders_dict["in_val"]
    }
    return deit3b_2layers_all_dataloaders_dict


def subset_dict_by_keys(input_dict, keys):
    return {
        key: input_dict[key] for key in keys
    }


def update_key_names(d, suffix):
    return {
        key + suffix: value
            for key, value
                in d.items()
    }


def prepare_deit3b_dataloaders(eval_type, cached_datasets_info):

    deit3b_2layers_all_dataloaders_dict = make_cached_dataloaders(
        subset_dict_by_keys(
            get_with_assert(cached_datasets_info, "path_mapping"),
            get_with_assert(cached_datasets_info, f"{eval_type}_keys")
        )
    )

    wrappers_dict = cached_datasets_info.get("wrappers", {})

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

        roc_auc = roc_auc_score(labels, metric_values)
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
    ] + [name for name, _ in metrics_mappings]

    if "mutual_information_per_sample" in metric_names:
        metric_names.append("inv_mutual_information_per_sample")

    if "average_max_logit_per_sample" in metric_names:
        metric_names.append("inv_average_max_logit_per_sample")

    res = {}

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
        metrics_mappings,
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
            save_to_pkl(res, save_path, results_key)

    if verbose:
        log_or_print(
            pretty_json(res),
            logger,
            auto_newline=True
        )
    save_to_pkl(res, save_path, results_key)
    return res


def extract_from_pkl(path, key=None, assert_non_empty=True):
    assert os.path.exists(path)
    res = torch.load(path)
    if key is not None:
        if assert_non_empty:
            assert key in res, f"{key} not found in {res}"
        res = res.get(key, {})
    if assert_non_empty:
        assert len(res) > 0
    return res


def save_to_pkl(data, path, key=None):
    pkl_storage = torch.load(path) if os.path.exists(path) else {}
    if key is not None:
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

    else:
        assert eval_type == "ood_det"
        metrics_mappings = [
            ("div_different_preds_per_sample", div_different_preds_per_sample),
            ("div_continous_unique_per_sample", div_continous_unique_per_sample)
        ]

        metrics_mappings += EXTENDED_METRICS_PER_SAMPLE

    return metrics_mappings
