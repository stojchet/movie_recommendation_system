import argparse
import os
from typing import Dict, Tuple, Callable

import numpy as np
import tensorflow as tf

from input.dataset.preprocess.dataset_with_trailer_data.trailer_dataset_path_util import \
    get_path_to_trailer_dataset_tf_data, get_path_to_trailer_dataset_tensor, get_base_trailer_dataset_path
from src.input.dataset.preprocess.dataset_with_trailer_data.trailer_process.process_all_trailers import ProcessAllTrailers
from src.input.dataset.preprocess.dataset_with_trailer_data.trailer_process.util.dimensionality_reduction import \
    DimensionalityReductor
from src.utils.constants import TRAIN, DEV, TEST
from src.input.dataset.util.util import DatasetUtil
from src.utils.utils import TRAILER_DATA, BASE_TENSOR_DATASET, \
    WITH_TRAILER

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument("--aggregation", default="max", type=str, help="Aggregation method - max or avg.")
parser.add_argument("--dimensionality_reduction", default="pca", type=str, help="Dimensionality reduction method - pca, k-pca or t-sne.")


# Note: dataset should not be batched - batch it before use
class DatasetWithTrailerData:
    __X: tf.Tensor
    __y: tf.Tensor

    def __init__(self, dataset_type: str, aggregation_type: str, dimensionality_reduction: str, base_dataset_path = BASE_TENSOR_DATASET):
        self.dataset_type: str = dataset_type
        self.aggregation_type: str = aggregation_type
        self.dimensionality_reduction = dimensionality_reduction
        self.base_dataset_path = get_base_trailer_dataset_path(self.aggregation_type, self.dimensionality_reduction)
        self.tf_data_dataset_path = get_path_to_trailer_dataset_tf_data(self.aggregation_type,
                                                                        self.dimensionality_reduction)
        self.tensor_dataset_path = get_path_to_trailer_dataset_tensor(self.aggregation_type, self.dimensionality_reduction)

        self.__X, self.__y = DatasetUtil.load_tensor_dataset(dataset_type, base_dataset_path)

        self.__create_directories()
        self.__create_and_save_dataset()

    def save_dataset(self, dataset: tf.data.Dataset):
        DatasetUtil.save_dataset(dataset=dataset,
                                 dataset_type=self.dataset_type,
                                 path=self.tf_data_dataset_path)

    def save_new_X_dataset_with_trailer_features(self,
                                                 base_features: tf.Tensor,
                                                 labels: tf.Tensor,
                                                 trailer_features: tf.Tensor):
        base_features = tf.concat([base_features, trailer_features], axis=-1)
        DatasetUtil.save_tensor_dataset_features(base_features, self.dataset_type, path=self.tensor_dataset_path)
        DatasetUtil.save_tensor_dataset_labels(labels, self.dataset_type, path=self.tensor_dataset_path)

    def __decrease_reference_counter(self):
        self.__X, self.__y = None, None

    def __create_directories(self):
        if not WITH_TRAILER.exists(): os.mkdir(WITH_TRAILER)
        if not self.base_dataset_path.exists(): os.mkdir(self.base_dataset_path)
        if not self.tensor_dataset_path.exists(): os.mkdir(self.tensor_dataset_path)
        if not self.tf_data_dataset_path.exists(): os.mkdir(self.tf_data_dataset_path)

    def __create_and_save_dataset(self):
        filtered_dataset, filtered_labels = self.__remove_pairs_that_have_no_trailer_data()
        trailer_features = self.__get_trailer_features_for_dataset(filtered_dataset=filtered_dataset)

        self.save_new_X_dataset_with_trailer_features(filtered_dataset, filtered_labels, trailer_features)
        dataset = self.__get_multiple_input_dataset_from_tensors_with_trailer_data(filtered_dataset,
                                                                                   filtered_labels,
                                                                                   trailer_features)

        self.save_dataset(dataset)

    def __remove_pairs_that_have_no_trailer_data(self) -> Tuple[tf.Tensor, tf.Tensor]:
        list_ids = [int(id) for id in set(ProcessAllTrailers.get_all_movies_that_have_trailers(TRAILER_DATA))]
        movie_ids_that_have_trailers = tf.convert_to_tensor(value=list_ids, dtype=tf.int32)

        indices_of_datapoints_to_keep = np.isin(self.__X[:, 0], movie_ids_that_have_trailers)
        filtered_dataset = self.__X[indices_of_datapoints_to_keep]
        filtered_labels = self.__y[indices_of_datapoints_to_keep]

        self.__decrease_reference_counter()
        return filtered_dataset, filtered_labels

    def __get_trailer_data(self) -> Dict[str, tf.Tensor]:
        return ProcessAllTrailers().get_all_trailer_features(self.aggregation_type)

    def __get_trailer_features_for_dataset(self, filtered_dataset) -> np.array:
        movie_ids = tf.cast(filtered_dataset[:, 0], tf.int32)
        trailer_features: Dict[str, tf.Tensor] = self.__get_trailer_data()

        tensor_features = tf.convert_to_tensor(list(trailer_features.values()))
        reduced_features = DimensionalityReductor().reduce_dimensions(self.dimensionality_reduction, tensor_features)
        reduced_features_dict = dict(zip(list(trailer_features.keys()), reduced_features))

        select_trailer_features: Callable[[np.bytes], np.array] \
            = lambda movie_id: reduced_features_dict[str(movie_id)]
        trailer_features_for_ids: np.array \
            = np.vectorize(select_trailer_features, signature='()->(n)')(movie_ids.numpy())
        return tf.convert_to_tensor(trailer_features_for_ids, dtype=tf.float32)

    @staticmethod
    def __get_multiple_input_dataset_from_tensors_with_trailer_data(X: tf.Tensor,
                                                                    y: tf.Tensor,
                                                                    trailer_features: tf.Tensor) -> tf.data.Dataset:
        y = tf.reshape(y, [-1])

        return tf.data.Dataset.from_tensor_slices(({"movie_ids": X[:, 0],
                                                    "user_ids": X[:, 1],
                                                    "movie_features": X[:, 2:26],
                                                    "user_context": X[:, -2:],
                                                    "trailer_features": trailer_features,
                                                    }, y))


if __name__ == "__main__":
    args = vars(parser.parse_args([] if "__file__" not in globals() else None))
    #
    # train = DatasetWithTrailerData(TRAIN, args["aggregation"], args["dimensionality_reduction"])
    # dev = DatasetWithTrailerData(DEV, args["aggregation"], args["dimensionality_reduction"])
    test = DatasetWithTrailerData(TEST,
                                  args["aggregation"],
                                  args["dimensionality_reduction"],
                                  get_path_to_trailer_dataset_tensor(args["aggregation"],
                                                                     args["dimensionality_reduction"]))
