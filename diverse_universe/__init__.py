import os
from stuned.utility.utils import PROJECT_ROOT_ENV_NAME
# import pkgutil
import importlib

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

os.environ[PROJECT_ROOT_ENV_NAME] = PROJECT_ROOT

# PATH_TO_MODEL_VS_HUMAN = os.path.join(
#     PROJECT_ROOT,
#     "envs",
#     "diverse_universe",
#     "lib",
#     "python3.10",
#     "site-packages"
# )
try:
    # mvh_package = pkgutil.get_loader("modelvshuman")
    mvh_package = importlib.util.find_spec("modelvshuman")
    # print(mvh_package)
except:
    raise ImportError("modelvshuman package not found")
# assert os.path.exists(PATH_TO_MODEL_VS_HUMAN), \
#     f"modelvshuman package not found in {PATH_TO_MODEL_VS_HUMAN}"
os.environ["MODELVSHUMANDIR"] = os.path.dirname(
    # os.path.dirname(mvh_package.get_filename())
    os.path.dirname(mvh_package.origin)
)

# from stuned.utility.imports import lazy_import

# # lazy imports
# tensorflow = lazy_import("tensorflow")  # otherwise slow imports
