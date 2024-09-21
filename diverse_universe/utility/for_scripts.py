import os
import sys
import torch
import numpy as np
from IPython.display import display
import pandas as pd
from stuned.utility.utils import (
    get_project_root_path,
    log_or_print,
    read_yaml
)


# local imports
sys.path.insert(
    0,
    os.path.dirname(os.path.abspath(''))
)
from diverse_universe.local_datasets.imagenet_c import (
    IN_C_DATALOADERS_NAMES,
)
from diverse_universe.cross_eval.eval import (
    extract_from_pkl
)
sys.path.pop(0)


DEFAULT_ROUND = 9
HYPERPARAM_PREFIX = "__hyperparam__"
ALL_CORRUPTION_NAMES = list(IN_C_DATALOADERS_NAMES.values())


def plot_table_from_cross_dict(
    path_to_dict,
    value_name,
    round_to=None,
    to_show=True,
    multiplier=1,
    merge_map=None,
    logger=None,
    model_names=None,
    dict_key=None
):
    def init_with_dicts(d, key):
        if key not in d:
            d[key] = {}

    def round_string(value, round_to=round_to, multiplier=multiplier, sep="+-"):
        mean, std = extract_mean_std(value, sep=sep)
        mean *= multiplier
        std *= multiplier
        return make_mean_std_str(mean, std, round_to, sep=sep)

    def as_sorted(iterable):
        return sorted(list(iterable), key=lambda x: x[0])

    def merge_dicts(dicts):
        res = {}
        for d in dicts:
            for key in d:
                if key not in res:
                    res[key] = d[key]
                else:
                    assert len(
                        set(d[key].keys()).intersection(set(res[key].keys()))
                    ) == 0
                    res[key] |= d[key]

        return res

    if isinstance(path_to_dict, list):
        # cross_dict = merge_dicts([torch.load(path) for path in path_to_dict])
        cross_dict = merge_dicts(
            [extract_from_pkl(path, key=dict_key) for path in path_to_dict]
        )
    else:
        # cross_dict = torch.load(path_to_dict)
        cross_dict = extract_from_pkl(path_to_dict, key=dict_key)

    res = {}
    hyperparams = {}
    # cross_dict: (i, j, k) model i -> dataset j -> stats k
    for model_name, model_dict in as_sorted(cross_dict.items()):

        init_with_dicts(res, model_name)
        init_with_dicts(hyperparams, model_name)

        if model_names is not None:
            assert isinstance(model_names, list)
            model_names.append(model_name)

        for dataset_name, stats in as_sorted(model_dict.items()):

            if HYPERPARAM_PREFIX in dataset_name:
                split = dataset_name.split(HYPERPARAM_PREFIX)
                assert len(split) == 2
                hyperparams[model_name][split[1]] = stats
                continue

            assert isinstance(stats, dict)
            if value_name not in stats:
                log_or_print(
                    f"For {model_name} on {dataset_name} value {value_name} "
                    f"not found among {stats.keys()}",
                    logger,
                    auto_newline=True
                )
            value = stats.get(value_name)

            if value is not None and round_to is not None:
                value = round_string(value)

            res[model_name][dataset_name] = value

        if merge_map is not None:
            # merge_map: merged_name -> [<names of datasets to merge>]
            for merged_dataset in merge_map.keys():
                datasets_to_merge_names = merge_map[merged_dataset]
                datasets_to_merge_values = []
                for dataset_to_merge in datasets_to_merge_names:
                    datasets_to_merge_values.append(
                        res[model_name].pop(dataset_to_merge)
                    )

                res[model_name][merged_dataset] = merge_mean_std(
                   datasets_to_merge_values,
                   round_to
                )

    df = pd.DataFrame.from_dict(res, orient='index')
    hyperparams = pd.DataFrame.from_dict(hyperparams, orient='index')

    if to_show:
        log_or_print(
            f"Table for {value_name}",
            logger,
            auto_newline=True
        )
        display(df)
    return df, hyperparams


def extract_mean_std(value, sep="+-"):
    if value is None:
        return None, None
    mean, std = map(float, value.split(sep))
    return mean, std

def make_mean_std_str(mean, std, round_to, sep="+-"):
    if mean is None or std is None:
        assert std is None and mean is None
        return None

    return f"{round(mean, round_to)} {sep} {round(std, round_to)}"


def merge_mean_std(values, round_to, sep="+-"):

    means, stds = [], []
    for value in values:
        mean, std = extract_mean_std(value, sep=sep)
        if mean is not None:
            means.append(mean)
        if std is not None:
            stds.append(std)

    res_mean, res_std = None, None
    if len(means) > 0:
        res_mean = np.array(means).mean()
    if len(stds) > 0:
        res_std = np.sqrt((np.array(stds) ** 2).sum())
    return make_mean_std_str(res_mean, res_std, round_to, sep=sep)


def get_long_table(
    pths,
    metric_names,
    axis,
    row_names,
    col_names,
    format_func,
    group_by_columns=False,
    merge_map=None,
    dict_key=None
):

    dfs = []
    hyperparams = None

    for metric_name in metric_names:

        if row_names is None:
            extracted_model_names = []
        else:
            extracted_model_names = None

        if isinstance(metric_name, tuple):
            assert len(metric_name) == 2
            multiplier = metric_name[1]
            metric_name = metric_name[0]
        else:
            multiplier = 1

        df, hyperparams_df = plot_table_from_cross_dict(
            pths,
            metric_name,
            round_to=DEFAULT_ROUND,
            to_show=False,
            multiplier=multiplier,
            merge_map=merge_map,
            model_names=extracted_model_names,
            dict_key=dict_key
        )
        if hyperparams is None:
            hyperparams = hyperparams_df
        else:
            assert hyperparams.equals(hyperparams_df)

        dfs.append(df)

        if extracted_model_names is not None:
            row_names = extracted_model_names

    return merge_dfs(
        dfs,
        axis,
        row_names=row_names,
        col_names=col_names,
        format_func=format_func,
        group_by_columns=group_by_columns,
        metric_names=metric_names,
        just_concat=[hyperparams_df]
    )


def merge_dfs(
    dfs,
    axis,
    row_names=None,
    col_names=None,
    format_func=None,
    group_by_columns=False,
    insert_empty_cols=True,
    metric_names=None,
    just_concat=[]
):

    def insert_row_above(df, row_as_list):

        df = pd.concat(
            [pd.DataFrame([row_as_list], columns=df.columns), df]
        )
        return df

    if metric_names is not None:
        assert len(metric_names) == len(dfs)

    if group_by_columns:
        pre_groupped = {}
    else:
        pre_groupped = None
    for i in range(len(dfs)):
        if row_names is not None:
            dfs[i] = dfs[i].loc[row_names]
        if col_names is not None:
            dfs[i] = dfs[i].loc[:, col_names]
        if format_func is not None:
            dfs[i] = dfs[i].applymap(format_func)

        if metric_names is not None:
            row_as_list = [metric_names[i]] * len(dfs[i].columns)
            dfs[i] = insert_row_above(
                dfs[i],
                row_as_list
            )

        if pre_groupped is not None:
            for column_name in dfs[i].columns:
                if column_name not in pre_groupped:
                    pre_groupped[column_name] = []

                pre_groupped[column_name].append(
                    dfs[i][[column_name]]
                )

    if pre_groupped is None:
        dfs_to_concat = dfs
    else:
        dfs_to_concat = [
                pd.concat(
                    [
                        single_column
                            for single_column
                                in pre_groupped_for_column
                    ],
                    axis=axis
                )
                        for pre_groupped_for_column
                        in pre_groupped.values()
            ]

    # to have empty row on top and be well aligned with other dfs
    for i in range(len(just_concat)):
        just_concat[i] = insert_row_above(
            just_concat[i],
            ["hyperparam"] * len(just_concat[i].columns)
        )

    dfs_to_concat = just_concat + dfs_to_concat
    if insert_empty_cols:
        dfs_with_inserted = []
        for i, df in enumerate(dfs_to_concat):
            dfs_with_inserted.append(df)
            if i + 1 != len(dfs_to_concat):
                dfs_with_inserted.append(pd.Series(dtype='int'))
        dfs_to_concat = dfs_with_inserted

    res_df = pd.concat(dfs_to_concat, axis=axis)
    return res_df


def format_number(pm_str):
    if pm_str is None:
        return pm_str
    number = pm_str.split('+-')[0]
    return abs(float(number))
