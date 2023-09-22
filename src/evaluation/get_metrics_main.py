import argparse
import json
import os

from evaluation.accuracy_error_methods import AccuracyAndErrorBasedMethods
from evaluation.general_preformance_methods import GeneralPerformanceMetrics
from evaluation.ranking_based_metrics import RankingBasedMetrics
from evaluation.predictions_util import ProcessPredictions
from model.core.recommendation_system_core import RecommendationSystemCore
from utils.utils import METRICS_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--include_trailers", default=False, type=bool,
                    help="Do you want to get the dataset that has trailer features")
parser.add_argument("--aggregation_method", default="max", type=str,
                    help="Method for aggregating features extracted form keyframes (min, max, avg)")
parser.add_argument("--feature_reduction", default=None, type=str,
                    help="Method for feature reduction of trailer features (pca, t-sne k-pca)")
parser.add_argument("--filtered_dataset", default=False, type=bool,
                    help="Use filtered base dataset by removing those datapoints that have no trailer")
parser.add_argument("--model_name", default=None, type=str, help="Name of the saved model")


def get_path(config):
    if config["include_trailers"] is False:
        base = METRICS_PATH.joinpath("base")
        return base.joinpath("filtered") if config["filtered_dataset"] else base.joinpath("whole")
    return METRICS_PATH.joinpath("with_trailer", f"{config['aggregation_method']}_{config['feature_reduction']}")


def get_metrics_summary(config):
    predictions = RecommendationSystemCore.load_predictions(config)
    predictions_process = ProcessPredictions(predictions=predictions,
                                             config=config,
                                             include_trailers=config["include_trailers"])

    path = get_path(config)

    eval = RankingBasedMetrics(predictions_process)
    summary = eval.get_metrics_summary()
    summary = summary_Serializable(summary)
    if not path.exists(): os.mkdir(path)

    with open(path.joinpath("ranking_metrics.json"), "w") as file:
        json.dump(summary, file)

    eval = GeneralPerformanceMetrics(predictions_process)
    summary = eval.get_metrics_summary()
    summary = summary_Serializable(summary)
    with open(path.joinpath("general_performance_metrics.json"), "w") as file:
        json.dump(summary, file)

    eval = AccuracyAndErrorBasedMethods(predictions_process)
    summary = eval.get_summary()
    summary = summary_Serializable(summary)
    with open(path.joinpath("accuracy_metrics.json"), "w") as file:
        json.dump(summary, file)


def summary_Serializable(summary):
    for k, c in summary.items():
        summary[k] = str(c)

    return summary


if __name__ == "__main__":
    config = vars(parser.parse_args([] if "__file__" not in globals() else None))
    get_metrics_summary(config)
