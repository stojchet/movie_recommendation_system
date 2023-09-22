import pathlib

PROJECT_DIR = pathlib.Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_DIR.joinpath("data")

BASE = DATA_DIR.joinpath("base")
BASE_TENSOR_DATASET = BASE.joinpath("tensor")
BASE_TF_DATA = BASE.joinpath("tf_data")
BASE_FILTERED = BASE.joinpath("filtered")

WITH_TRAILER = DATA_DIR.joinpath("with_trailer")
WITH_TRAILER_TENSOR_DATASET = WITH_TRAILER.joinpath("tensor")
WITH_TRAILER_TF_DATA = WITH_TRAILER.joinpath("tf_data")

WHOLE_TENSOR_DATASET = DATA_DIR.joinpath("whole", "tensor")
WHOLE_TF_DATA_DATASET = DATA_DIR.joinpath("whole", "tf_data")

MODEL_INFO_DIR = PROJECT_DIR.joinpath("model_info")
METRICS_PATH = PROJECT_DIR.joinpath("metrics")

TRAILER_DATA = PROJECT_DIR.joinpath("trailer_data")
MOVIELENS_PATH = PROJECT_DIR.joinpath("ml-25m")
MOVIELENS_MOVIES_PATH = MOVIELENS_PATH.joinpath("movies.csv")
MOVIELENS_RATINGS_PATH = MOVIELENS_PATH.joinpath("ratings.csv")
YOUTUBE_DATA = PROJECT_DIR.joinpath("ml-20m-youtube")

PATH_TO_MODEL = PROJECT_DIR.joinpath("model", "recommendation_system_core", "my_model.h5")
PATH_TO_TRAILER_FEATURES = PROJECT_DIR.joinpath("trailer_data")
