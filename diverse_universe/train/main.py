import os
import sys
from stuned.utility.helpers_for_main import prepare_wrapper_for_experiment


# local modules
# sys.path.pop(0)  # to avoid collisions with "configs.py" inside train_eval folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from diverse_universe.train.train import (
    train_eval_loop
)
from diverse_universe.train.configs import (
    check_exp_config,
    patch_exp_config
)
# from utility.helpers_for_main import prepare_wrapper_for_experiment
sys.path.pop(0)


def main():
    prepare_wrapper_for_experiment(
        check_exp_config,
        patch_exp_config
    )(
        train_eval_loop
    )()


if __name__ == "__main__":
    main()
