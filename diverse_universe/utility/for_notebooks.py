import torch
import os
import sys
import numpy as np
# import pandas as pd
# import random
# from IPython.display import display
from stuned.utility.logger import log_or_print
from stuned.utility.utils import (
    get_project_root_path,
    append_dict,
    pretty_json
)


# local imports
sys.path.insert(
    0,
    # os.path.join(
    #     os.path.dirname(os.path.abspath('')), "src"
    # )
    get_project_root_path()
)
from diverse_universe.local_datasets.utils import (
    evaluate_model,
    evaluate_ensemble
)
# from utility.utils import (
#     pretty_json,
#     append_dict,
#     extract_list_from_huge_string
# )
from diverse_universe.local_models.common import (
    # make_ensemble_from_model_list,
    get_model,
    make_ensembles
)
from diverse_universe.local_models.wrappers import (
    # make_ensemble_from_model_list,
    wrap_model
)
# from utility.logger import (
#     log_or_print
# )
# from train_eval.models import (
#     is_ensemble
# )
from diverse_universe.local_models.ensemble import (
    # REDNECK_ENSEMBLE_KEY,
    # SINGLE_MODEL_KEY,
    # POE_KEY,
    is_ensemble,
    # make_redneck_ensemble,
    # split_linear_layer
)
from diverse_universe.train_eval import (
    METRICS
)
sys.path.pop(0)


# DEFAULT_ROUND = 9
# HYPERPARAM_PREFIX = "__hyperparam__"


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
    metrics_mappings=METRICS,
    prune_metrics=["dis"],
    evaluation_func=None,
    evaluation_kwargs={},
    save_after_each_model=True,
    logger=None,
    model_to_prop_dict=None
):

    if os.path.exists(save_path):
        log_or_print(
            f"Loading existing results from {save_path}",
            logger,
            auto_newline=True
        )
        res = torch.load(save_path)
    else:
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
            torch.save(res, save_path)

    if verbose:
        log_or_print(
            pretty_json(res),
            logger,
            auto_newline=True
        )
    torch.save(res, save_path)
    return res


# def extract_mean_std(value, sep="+-"):
#     if value is None:
#         return None, None
#     mean, std = map(float, value.split(sep))
#     return mean, std


# def make_mean_std_str(mean, std, round_to, sep="+-"):
#     if mean is None or std is None:
#         assert std is None and mean is None
#         return None

#     return f"{round(mean, round_to)} {sep} {round(std, round_to)}"


# def plot_table_from_cross_dict(
#     path_to_dict,
#     value_name,
#     round_to=None,
#     to_show=True,
#     multiplier=1,
#     merge_map=None,
#     logger=None,
#     model_names=None
# ):
#     def init_with_dicts(d, key):
#         if key not in d:
#             d[key] = {}

#     def round_string(value, round_to=round_to, multiplier=multiplier, sep="+-"):
#         mean, std = extract_mean_std(value, sep=sep)
#         mean *= multiplier
#         std *= multiplier
#         return make_mean_std_str(mean, std, round_to, sep=sep)

#     def as_sorted(iterable):
#         return sorted(list(iterable), key=lambda x: x[0])

#     def merge_dicts(dicts):
#         res = {}
#         for d in dicts:
#             for key in d:
#                 if key not in res:
#                     res[key] = d[key]
#                 else:
#                     assert len(
#                         set(d[key].keys()).intersection(set(res[key].keys()))
#                     ) == 0
#                     res[key] |= d[key]

#         return res

#     if isinstance(path_to_dict, list):
#         cross_dict = merge_dicts([torch.load(path) for path in path_to_dict])
#     else:
#         cross_dict = torch.load(path_to_dict)

#     res = {}
#     hyperparams = {}
#     # cross_dict: (i, j, k) model i -> dataset j -> stats k
#     for model_name, model_dict in as_sorted(cross_dict.items()):

#         init_with_dicts(res, model_name)
#         init_with_dicts(hyperparams, model_name)

#         if model_names is not None:
#             assert isinstance(model_names, list)
#             model_names.append(model_name)

#         for dataset_name, stats in as_sorted(model_dict.items()):

#             if HYPERPARAM_PREFIX in dataset_name:
#                 split = dataset_name.split(HYPERPARAM_PREFIX)
#                 assert len(split) == 2
#                 hyperparams[model_name][split[1]] = stats
#                 continue

#             assert isinstance(stats, dict)
#             if value_name not in stats:
#                 log_or_print(
#                     f"For {model_name} on {dataset_name} value {value_name} "
#                     f"not found among {stats.keys()}",
#                     logger,
#                     auto_newline=True
#                 )
#             value = stats.get(value_name)

#             if value is not None and round_to is not None:
#                 value = round_string(value)

#             res[model_name][dataset_name] = value

#         if merge_map is not None:
#             # merge_map: merged_name -> [<names of datasets to merge>]
#             for merged_dataset in merge_map.keys():
#                 datasets_to_merge_names = merge_map[merged_dataset]
#                 datasets_to_merge_values = []
#                 for dataset_to_merge in datasets_to_merge_names:
#                     datasets_to_merge_values.append(
#                         res[model_name].pop(dataset_to_merge)
#                     )

#                 res[model_name][merged_dataset] = merge_mean_std(
#                    datasets_to_merge_values,
#                    round_to
#                 )

#     df = pd.DataFrame.from_dict(res, orient='index')
#     hyperparams = pd.DataFrame.from_dict(hyperparams, orient='index')

#     if to_show:
#         log_or_print(
#             f"Table for {value_name}",
#             logger,
#             auto_newline=True
#         )
#         display(df)
#     return df, hyperparams


# def merge_mean_std(values, round_to, sep="+-"):
#     means, stds = [], []
#     for value in values:
#         mean, std = extract_mean_std(value, sep=sep)
#         if mean is not None:
#             means.append(mean)
#         if std is not None:
#             stds.append(std)

#     res_mean, res_std = None, None
#     if len(means) > 0:
#         res_mean = np.array(means).mean()
#     if len(stds) > 0:
#         res_std = np.sqrt((np.array(stds) ** 2).sum())
#     return make_mean_std_str(res_mean, res_std, round_to, sep=sep)


def evaluate_ensembles(
    ensembles,
    dataloader,
    device=torch.device("cuda:0"),
    feature_extractor=None,
    wrap=None,
    metrics_mappings=METRICS,
    prune_metrics=["dis"],
    evaluation_func=None,
    evaluation_kwargs={},
    logger=None
):

    if isinstance(ensembles[0], str):
        ensemble_paths = ensembles
        ensembles = [get_model(ensemble_path) for ensemble_path in ensemble_paths]

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
    metrics_mappings=METRICS,
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


# def make_ensembles_from_paths(paths, group_by, num_ensembles, base_model=None):
#     paths = extract_list_from_huge_string(paths)
#     total_paths = len(paths)
#     assert total_paths >= num_ensembles
#     assert total_paths >= group_by

#     indices_per_ensemble = []

#     # select indices
#     for j in range(num_ensembles):
#         indices_per_ensemble.append(
#             set(random.sample(list(range(total_paths)), group_by))
#         )

#     res = [[] for _ in range(num_ensembles)]

#     for i, path in enumerate(paths):
#         for j in range(len(indices_per_ensemble)):
#             if i in indices_per_ensemble[j]:
#                 res[j].append(get_model(path, base_model=base_model))

#     return [make_ensemble_from_model_list(model_list) for model_list in res]


# def format_number(pm_str):
#     if pm_str is None:
#         return pm_str
#     number = pm_str.split('+-')[0]
#     return abs(float(number))


# def insert_row_above(df, row_as_list):

#     df = pd.concat(
#         [pd.DataFrame([row_as_list], columns=df.columns), df]
#     )
#     return df


# def merge_dfs(
#     dfs,
#     axis,
#     row_names=None,
#     col_names=None,
#     format_func=None,
#     group_by_columns=False,
#     insert_empty_cols=True,
#     metric_names=None,
#     just_concat=[]
# ):

#     if metric_names is not None:
#         assert len(metric_names) == len(dfs)

#     if group_by_columns:
#         pre_groupped = {}
#     else:
#         pre_groupped = None
#     for i in range(len(dfs)):
#         if row_names is not None:
#             dfs[i] = dfs[i].loc[row_names]
#         if col_names is not None:
#             dfs[i] = dfs[i].loc[:, col_names]
#         if format_func is not None:
#             dfs[i] = dfs[i].applymap(format_func)

#         if metric_names is not None:
#             row_as_list = [metric_names[i]] * len(dfs[i].columns)
#             dfs[i] = insert_row_above(
#                 dfs[i],
#                 row_as_list
#             )

#         if pre_groupped is not None:
#             for column_name in dfs[i].columns:
#                 if column_name not in pre_groupped:
#                     pre_groupped[column_name] = []

#                 pre_groupped[column_name].append(
#                     dfs[i][[column_name]]
#                 )

#     if pre_groupped is None:
#         dfs_to_concat = dfs
#     else:
#         dfs_to_concat = [
#                 pd.concat(
#                     [
#                         single_column
#                             for single_column
#                                 in pre_groupped_for_column
#                     ],
#                     axis=axis
#                 )
#                         for pre_groupped_for_column
#                         in pre_groupped.values()
#             ]

#     # to have empty row on top and be well aligned with other dfs
#     for i in range(len(just_concat)):
#         just_concat[i] = insert_row_above(
#             just_concat[i],
#             ["hyperparam"] * len(just_concat[i].columns)
#         )

#     dfs_to_concat = just_concat + dfs_to_concat
#     if insert_empty_cols:
#         dfs_with_inserted = []
#         for i, df in enumerate(dfs_to_concat):
#             dfs_with_inserted.append(df)
#             if i + 1 != len(dfs_to_concat):
#                 dfs_with_inserted.append(pd.Series(dtype='int'))
#         dfs_to_concat = dfs_with_inserted

#     res_df = pd.concat(dfs_to_concat, axis=axis)
#     return res_df


# def get_long_table(
#     pths,
#     metric_names,
#     axis,
#     row_names,
#     col_names,
#     format_func,
#     group_by_columns=False,
#     merge_map=None
# ):

#     dfs = []
#     hyperparams = None

#     for metric_name in metric_names:

#         if row_names is None:
#             extracted_model_names = []
#         else:
#             extracted_model_names = None

#         if isinstance(metric_name, tuple):
#             assert len(metric_name) == 2
#             multiplier = metric_name[1]
#             metric_name = metric_name[0]
#         else:
#             multiplier = 1

#         df, hyperparams_df = plot_table_from_cross_dict(
#             pths,
#             metric_name,
#             round_to=DEFAULT_ROUND,
#             to_show=False,
#             multiplier=multiplier,
#             merge_map=merge_map,
#             model_names=extracted_model_names
#         )
#         if hyperparams is None:
#             hyperparams = hyperparams_df
#         else:
#             assert hyperparams.equals(hyperparams_df)

#         dfs.append(df)

#         if extracted_model_names is not None:
#             row_names = extracted_model_names

#     return merge_dfs(
#         dfs,
#         axis,
#         row_names=row_names,
#         col_names=col_names,
#         format_func=format_func,
#         group_by_columns=group_by_columns,
#         metric_names=metric_names,
#         just_concat=[hyperparams_df]
#     )


# def make_models_dict_from_huge_string(huge_string, keys, id_prefix=""):
#     # TODO(Alex |02.04.2024): maybe later we can extend it to the whole gsheet
#     # and update it by adding new columns

#     def make_id(id_prefix, keys, split):
#         id_name = id_prefix
#         assert "path" in keys
#         path = None
#         properties = {}
#         for i, (key, value) in enumerate(zip(keys, split)):
#             if key != "path":
#                 id_name += key + '_' + value
#                 if i + 1 != len(keys):
#                     id_name += '_'
#                 properties[HYPERPARAM_PREFIX + key] = value
#             else:
#                 path = value
#         return id_name, path, properties

#     res = {}
#     name_to_prop = {}

#     huge_string = huge_string.replace('\t', ' ').replace('\n', ' ')

#     split = huge_string.split()

#     if keys is None:
#         assert id_prefix != ""
#         return {id_prefix: split}, {id_prefix: {}}

#     assert len(split) % len(keys) == 0, \
#             f"split {split} is not suitable for keys {keys}"

#     current_tuple = []
#     for item in split:

#         current_tuple.append(item)
#         if len(current_tuple) < len(keys):
#             continue

#         id_name, path, properties = make_id(id_prefix, keys, current_tuple)

#         if id_name in name_to_prop:
#             assert name_to_prop[id_name] == properties
#         else:
#             name_to_prop[id_name] = properties

#         assert path is not None

#         append_dict(res, {id_name: path}, allow_new_keys=True)

#         current_tuple = []

#     return res, name_to_prop


# def ensemble_from_multiple_paths(
#     save_path,
#     paths_as_long_string=None,
#     base_model=None,
#     group_by=None
# ):
#     if group_by is None:
#         raise ValueError("group_by must be specified")
#     if not os.path.exists(save_path):
#         assert paths_as_long_string is not None
#         ensemble = make_ensembles_from_paths(
#             paths_as_long_string,
#             group_by=group_by,
#             num_ensembles=1,
#             base_model=base_model
#         )[0]
#         torch.save(ensemble, save_path)
#     return save_path
