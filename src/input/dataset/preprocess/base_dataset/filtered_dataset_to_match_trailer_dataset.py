import os
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from input.dataset.preprocess.dataset_with_trailer_data.trailer_process.process_all_trailers import ProcessAllTrailers
from input.dataset.util.util import DatasetUtil
from utils.constants import TRAIN, DEV, TEST
from utils.utils import BASE_TENSOR_DATASET, TRAILER_DATA, BASE_FILTERED


# Get only datapoints that have trailers - then train on the filtered dataset
class FilteredDatasetToMatchTrailerDataset:
    def __init__(self, aggregation_type: str = "avg"):
        self.aggregation_type = aggregation_type
        self.__X, self.__y = DatasetUtil.load_tensor_dataset("whole", BASE_TENSOR_DATASET)
        self.trailer_features = self.__get_trailer_data()
        self.filter_and_save()

    def __get_trailer_data(self) -> Dict[str, tf.Tensor]:
        return ProcessAllTrailers().get_all_trailer_features(self.aggregation_type)

    def __decrease_reference_counter(self):
        self.__X, self.__y = None, None

    @staticmethod
    def __create_directories():
        if not BASE_FILTERED.exists(): os.mkdir(BASE_FILTERED)

    def __remove_pairs_that_have_no_trailer_data(self) -> Tuple[tf.Tensor, tf.Tensor]:
        list_ids = [int(id) for id in set(ProcessAllTrailers.get_all_movies_that_have_trailers(TRAILER_DATA))]
        movie_ids_that_have_trailers = tf.convert_to_tensor(value=list_ids, dtype=tf.int32)

        indices_of_datapoints_to_keep = np.isin(self.__X[:, 0], movie_ids_that_have_trailers)
        filtered_dataset = self.__X[indices_of_datapoints_to_keep]
        filtered_labels = self.__y[indices_of_datapoints_to_keep]

        self.__decrease_reference_counter()
        return filtered_dataset, filtered_labels

    @staticmethod
    def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                                  shuffle_size=10000):
        assert (train_split + test_split + val_split) == 1

        if shuffle:
            # Specify seed to always have the same split distribution between runs
            ds = ds.shuffle(shuffle_size, seed=12)

        train_size = int(train_split * ds_size)
        val_size = int(val_split * ds_size)

        train_ds = ds.take(train_size)
        val_ds = ds.skip(train_size).take(val_size)
        test_ds = ds.skip(train_size).skip(val_size)

        return train_ds, val_ds, test_ds

    def filter_and_save(self):
        X, y = self.__remove_pairs_that_have_no_trailer_data()
        y = tf.reshape(y, [-1])

        dataset = tf.data.Dataset.from_tensor_slices(({"movie_ids": X[:, 0],
                                                    "user_ids": X[:, 1],
                                                    "movie_features": X[:, 2:-2],
                                                    "user_context": X[:, -2:],
                                                    }, y))

        train_ds, val_ds, test_ds = self.get_dataset_partitions_tf(dataset, X.shape[0])
        DatasetUtil.save_datasets({
            TRAIN: train_ds,
            DEV: val_ds,
            TEST: test_ds,
        }, path=BASE_FILTERED)
        DatasetUtil.save_dataset(dataset, "whole", BASE_FILTERED)


if __name__ == "__main__":
    FilteredDatasetToMatchTrailerDataset()
