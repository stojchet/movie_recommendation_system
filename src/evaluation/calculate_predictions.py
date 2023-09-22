import argparse

import numpy as np

from evaluation.accuracy_error_methods import AccuracyAndErrorBasedMethods
from evaluation.general_preformance_methods import GeneralPerformanceMetrics
from evaluation.ranking_based_metrics import RankingBasedMetrics
from evaluation.predictions_util import ProcessPredictions
from input.dataset.prepared.dataset import Dataset
from input.dataset.util.util import DatasetUtil
from model.core.recommendation_system_core import RecommendationSystemCore
from utils.constants import TEST
from utils.utils import METRICS_PATH, BASE_FILTERED

parser = argparse.ArgumentParser()
parser.add_argument("--include_trailers", default=False, type=bool,
                    help="Do you want to get the dataset that has trailer features")
parser.add_argument("--aggregation_method", default="max", type=str,
                    help="Method for aggregating features extracted form keyframes (min, max, avg)")
parser.add_argument("--feature_reduction", default=None, type=str,
                    help="Method for feature reduction of trailer features (pca, t-sne k-pca)")
parser.add_argument("--filtered_dataset", default=False, type=bool,
                    help="Use filtered base dataset by removing those datapoints that have no trailer")

parser.add_argument("--preactivation_layers", default=3, type=int,
                    help="Preactivation layers used in recommendation system.")
parser.add_argument("--dropout", default=0.2, type=float, help="Full preactivation layer dropout.")
parser.add_argument("--dense_units", default=256, type=int,
                    help="Full preactivation layer dense layers number of units.")
parser.add_argument("--model_name", default=None, type=str, help="Name of the saved model")
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")


def get_path(config):
    if config["include_trailers"] is False:
        base = METRICS_PATH.joinpath("base")
        return base.joinpath("filtered") if config["filtered_dataset"] else base.joinpath("whole")
    return METRICS_PATH.joinpath("with_trailer",
                                 f"{config['aggregation_method']}_{config['dimensionality_reduction_method']}")


def get_and_save_predictions(config) -> np.ndarray[float]:
    dataset_config = DatasetUtil.load_config()
    model = RecommendationSystemCore.load(config=config, dataset_config=dataset_config)
    test = get_datasets(TEST,
                        config["include_trailers"],
                        config["aggregation_method"],
                        config["feature_reduction"],
                        config["filtered_dataset"]) \
        .batch(config["batch_size"], drop_remainder=False)
    return model.get_predictions(test, config=config)


def get_datasets(dataset_type: str,
                 include_trailers: bool,
                 aggregation_method: str,
                 dimensionality_reduction: str,
                 filtered_dataset: str):
    return Dataset(dataset_type, include_trailers, aggregation_method, dimensionality_reduction, filtered_dataset).get_dataset()


if __name__ == "__main__":
    config = vars(parser.parse_args([] if "__file__" not in globals() else None))
    get_and_save_predictions(config)
