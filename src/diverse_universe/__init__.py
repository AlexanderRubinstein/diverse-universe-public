import os
from stuned.utility.utils import PROJECT_ROOT_ENV_NAME

os.environ[PROJECT_ROOT_ENV_NAME] = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
