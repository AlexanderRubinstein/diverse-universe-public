import os
import sys

from stuned.utility.utils import (
    get_project_root_path,
)


# local imports
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(__file__))
)
from diverse_universe.utility.for_scripts import (
    ALL_CORRUPTION_NAMES,
    get_long_table,
    format_number
)
sys.path.pop(0)


PICKLES_PATH = os.path.join(get_project_root_path(), "result_pickles")


def main():

    results_path = os.path.join(PICKLES_PATH, "eval_results.pkl")
    assert os.path.exists(results_path), \
        f"Results are not found at the expected path: {results_path}"

    in_c_merge_map = {
        "C-1": [corruption + "_1" for corruption in ALL_CORRUPTION_NAMES],
        "C-5": [corruption + "_5" for corruption in ALL_CORRUPTION_NAMES]
    }

    # possible ensemble aggregation methods:
    # [
    #     "submodel_0",
    #     "submodel_1",
    #     "submodel_2",
    #     "submodel_3",
    #     "submodel_4",
    #     "mean_single_model",
    #     "best_single_model",
    #     "ensemble",
    #     "soup",
    #     "div_different_preds",
    #     "div_continous_unique",
    # ]

    ood_gen_df = get_long_table(
        [
            results_path,
        ],
        # ensemble aggregation methods
        [
            ("best_single_model", 100),
            ("ensemble", 100),
            ("soup", 100),
            "div_different_preds",
            "div_continous_unique",
        ],
        axis=1,
        row_names=None,
        col_names=[
            "in_val",
            "imagenet_a",
            "imagenet_r",
            "C-1",
            "C-5",
            "iNaturalist",
            "OpenImages"
        ],
        format_func=format_number,
        merge_map=in_c_merge_map,
        dict_key="ood_gen"
    )

    # possible uncertainty scores:
    #   [
    #     'ensemble_conf',
    #     'mean_submodel_conf',
    #     'div_different_preds_per_sample',
    #     'div_continous_unique_per_sample',
    #     'ens_entropy_per_sample',
    #     'average_entropy_per_sample',
    #     'mutual_information_per_sample',
    #     'average_energy_per_sample',
    #     'average_max_logit_per_sample',
    #     'a2d_score_per_sample',
    #     'inv_mutual_information_per_sample',
    #     'inv_average_max_logit_per_sample'
    # ]
    tables_folder = os.path.join(get_project_root_path(), "result_tables")
    os.makedirs(tables_folder, exist_ok=True)
    ood_gen_path = os.path.join(tables_folder, "ood_gen_df.csv")
    ood_gen_df.to_csv(ood_gen_path)
    print(f"OOD generalization results table is saved at: {ood_gen_path}")

    ood_det_df = get_long_table(
        [
            results_path,
        ],
        # uncertainty scores
        [
            'ensemble_conf',
            'div_continous_unique_per_sample'
        ],
        axis=1,
        row_names=None,
        col_names=[
            "C-1",
            "C-5",
            "iNaturalist",
            "OpenImages"
        ],
        format_func=format_number,
        merge_map=in_c_merge_map,
        group_by_columns=True,
        dict_key="ood_det"
    )
    ood_det_path = os.path.join(tables_folder, "ood_det_df.csv")
    print(f"OOD detection results table is saved at: {ood_det_path}")
    ood_det_df.to_csv(ood_det_path)


if __name__ == "__main__":
    main()
