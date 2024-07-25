import os
import sys
from stuned.utility.helpers_for_main import prepare_wrapper_for_experiment
# from stuned.utility.utils import (
#     get_project_root_path,
#     # append_dict,
#     # pretty_json
# )


# local modules
# sys.path.pop(0)  # to avoid collisions with "configs.py" inside train_eval folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from diverse_universe.cross_eval.eval import (
    cross_eval,
    check_eval_config,
    patch_eval_config
)
# from utility.helpers_for_main import prepare_wrapper_for_experiment
sys.path.pop(0)


def main():
    prepare_wrapper_for_experiment(
        check_eval_config,
        patch_eval_config
    )(
        cross_eval
    )()


if __name__ == "__main__":
    main()
