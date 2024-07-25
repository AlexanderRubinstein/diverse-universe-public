import os
from stuned.utility.utils import PROJECT_ROOT_ENV_NAME

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

os.environ[PROJECT_ROOT_ENV_NAME] = PROJECT_ROOT

PATH_TO_MODEL_VS_HUMAN = os.path.join(
    PROJECT_ROOT,
    "envs",
    "diverse_universe",
    "lib",
    "python3.10",
    "site-packages"
)
assert os.path.exists(PATH_TO_MODEL_VS_HUMAN), \
    f"modelvshuman package not found in {PATH_TO_MODEL_VS_HUMAN}"
os.environ["MODELVSHUMANDIR"] = PATH_TO_MODEL_VS_HUMAN
