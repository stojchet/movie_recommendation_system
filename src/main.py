#!/usr/bin/env python3
import argparse
from typing import Dict

from src.model.core.recommendation_system import RecommendationSystem

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--preactivation_layers", default=3, type=int,
                    help="Preactivation layers used in recommendation system.")
parser.add_argument("--dropout", default=0.2, type=float, help="Full preactivation layer dropout.")
parser.add_argument("--dense_units", default=256, type=int,
                    help="Full preactivation layer dense layers number of units.")
parser.add_argument("--model_name", default=None, type=str, help="Name of the saved model")

# Dataset Arguments
parser.add_argument("--include_trailers", default=False, type=bool,
                    help="Do you want to get the dataset that has trailer features")
parser.add_argument("--aggregation_method", default="max", type=str,
                    help="Method for aggregating features extracted form keyframes (min, max, avg)")
parser.add_argument("--feature_reduction", default="pca", type=str,
                    help="Method for feature reduction of trailer features (pca)")
parser.add_argument("--filtered_dataset", default=True, type=bool,
                    help="Use filtered base dataset by removing those datapoints that have no trailer")

# Other arguments
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--from_model", default=False, type=bool, help="Test using saved model.")


def main() -> None:
    rec_sys_train()


def get_test_hyperparameters(args: Dict[str, float], args_to_modify: Dict[str, float]):
    args.update(args_to_modify)
    return args


def rec_sys_train():
    args = parser.parse_args([] if "__file__" not in globals() else None)
    config = vars(args)
    model = RecommendationSystem(config=config)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main()

# relevant links
'''
https://github.com/chenxd2/MovieLens-Hybrid-Movie-Recommendation-System/blob/main/NCF-model.ipynb
https://www.tensorflow.org/recommenders/examples/basic_retrieval
https://blog.paperspace.com/movie-recommender-tensorflow/
'''

# TODO: save user movie max id, num ids for train dev test; add shapes form trailers
# TODO: set up a better pipeline for testing - I guess based on arguments or a config file

# TODO: collect datasets for testing - also maybe add some names based on agg and dim red
# TODO: add directory creation for trailer data.

# TODO save model in 2 folders - with and without trailer then just add the agg and dim red methods in the name.
#  Maybe even for those add folders
#  Do the same for predictions as well

# TODO: remove all trailers that have an error for downloading
#  Split rest in a few files and create multiple notebooks

# Maybe some videos are long - check if that's the case and remove them

# TODO:
#  DONE: 1. Run metrics on model
#  DONE: 2. Collect dataset with trailers.
#  3. Run new model on datalore - with creating branches
#  DONE: 4. modify ProcessPredictions in predictions_util.py to accept predictions as an argument
#  DONE: 5. Delete trailer features from git history
#  6. Fix extra metrics
#  Solution: run them on a small batch of examples

# TODO: https://www.tensorflow.org/ranking/tutorials/quickstart
#  Should I embed user and movie ids and names
#  check out: https://github.com/rposhala/Recommender-System-on-MovieLens-dataset/blob/main/Recommender_System_using_Softmax_DNN.ipynb

# {'mse': 1.5046651, 'rmse': 1.226648, 'mae': 1.0199077}