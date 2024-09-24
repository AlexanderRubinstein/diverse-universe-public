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
from diverse_universe.local_datasets.utils import (
    SHARING_SUFFIX,
    download_and_extract_tar
)
sys.path.pop(0)


CACHED_DATA_URL = "https://drive.google.com/file/d/1DT3uxqAt2-tvgYZAW5JIXnAjpiQ1FETq"
DATASETS_FOLDER = os.path.join(
    get_project_root_path(),
    "data",
    "datasets",
    "cached"
)


def main():
    parent_folder = os.path.dirname(DATASETS_FOLDER)
    os.makedirs(parent_folder, exist_ok=True)
    download_and_extract_tar(DATASETS_FOLDER, CACHED_DATA_URL + SHARING_SUFFIX)
    print("All cached data downloaded!")


if __name__ == "__main__":
    main()
