from typing import Tuple, List, Dict, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from src.input.dataset.preprocess.base_dataset.pd_data_frame_dataset import PdDataFrameDataset
from src.input.dataset.util.util import DatasetUtil
from src.utils.constants import TRAIN, DEV, TEST


class TfTensorDataset:
    def __init__(self, process_whole_dataset: bool, save_tensor_dataset: bool = True):
        self.__pd_datasets: Union[Dict[str, Union[Tuple[List[pd.DataFrame], pd.DataFrame], None]], None] = self.__get_pandas_dataset(process_whole_dataset)

        if process_whole_dataset:
            self.__get_and_save_main_dataset_in_tensor_form()
            self.__datasets = {"whole": self.__dataset_to_tensor_form("whole")}
            self.__pd_datasets = None
        else:
            if save_tensor_dataset:
                self.__get_and_save_dataset_in_tensor_form()

            self.__datasets: Dict[str, Tuple[List[tf.Tensor], tf.Tensor]] = self.__get_final_datasets_as_tensors()
            self.__pd_datasets = None
            self.__check_all_datasets_have_consistent_shapes()

    def get_dict_of_datasets(self) -> Dict[str, Tuple[List[tf.Tensor], tf.Tensor]]:
        return self.__datasets

    def __check_all_datasets_have_consistent_shapes(self):
        train = self.__datasets[TRAIN][0]
        dev = self.__datasets[DEV][0]
        test = self.__datasets[TEST][0]

        assert train[0].shape[1:] == dev[0].shape[1:] == test[0].shape[1:]
        assert train[1].shape[1:] == dev[1].shape[1:] == test[1].shape[1:]
        assert train[2].shape[1:] == dev[2].shape[1:] == test[2].shape[1:]
        assert train[3].shape[1:] == dev[3].shape[1:] == test[3].shape[1:]

    def __get_and_save_dataset_in_tensor_form(self):
        __tensor_datasets: Dict[str, Tuple[tf.Tensor, tf.Tensor]] = {}
        for dataset in [TRAIN, DEV, TEST]:
            X, y = self.__input_features_to_list_of_tensors(dataset), self.__labels_to_tensors(dataset)
            X = [tf.expand_dims(X[0], -1), tf.expand_dims(X[1], -1), X[2], X[3]]
            X = tf.concat(values=X, axis=-1)
            __tensor_datasets[dataset] = X, y

        DatasetUtil.save_all_tensor_datasets(__tensor_datasets)

    def __get_and_save_main_dataset_in_tensor_form(self):
        X, y = self.__input_features_to_list_of_tensors("whole"), self.__labels_to_tensors("whole")
        X = [tf.expand_dims(X[0], -1), tf.expand_dims(X[1], -1), X[2], X[3]]
        X = tf.concat(values=X, axis=-1)

        DatasetUtil.save_tensor_dataset_features(X, "whole")
        DatasetUtil.save_tensor_dataset_labels(y, "whole")

    @staticmethod
    def __get_pandas_dataset(process_whole_dataset: bool):
        pdDataFrameDataset = PdDataFrameDataset(process_whole_dataset)
        return pdDataFrameDataset.get_dict_of_datasets()

    def __get_final_datasets_as_tensors(self) -> Dict[str, Tuple[List[tf.Tensor], tf.Tensor]]:
        return {
            TRAIN: self.__dataset_to_tensor_form(TRAIN),
            DEV: self.__dataset_to_tensor_form(DEV),
            TEST: self.__dataset_to_tensor_form(TEST)
        }

    def __dataset_to_tensor_form(self, dataset: str) -> Tuple[List[tf.Tensor], tf.Tensor]:
        X, y = self.__input_features_to_list_of_tensors(dataset), self.__labels_to_tensors(dataset)
        self.__decrease_counter_for_dataset(dataset)
        return X, y

    def __input_features_to_list_of_tensors(self, dataset: str) -> List[tf.Tensor]:
        X_data, y_data = self.__pd_datasets[dataset]

        return [
            tf.convert_to_tensor(np.asarray(X_data[0]).astype('float32')),
            tf.convert_to_tensor(np.asarray(X_data[1]).astype('float32')),
            tf.convert_to_tensor(np.asarray(X_data[2]).astype('float32')),
            tf.convert_to_tensor(np.asarray(X_data[3]).astype('float32')),
        ]

    def __decrease_counter_for_dataset(self, dataset):
        self.__pd_datasets[dataset] = None

    def __labels_to_tensors(self, dataset: str) -> tf.Tensor:
        return tf.convert_to_tensor(np.asarray(self.__pd_datasets[dataset][1]).astype('float32'))


if __name__ == "__main__":
    TfTensorDataset(save_tensor_dataset=True, process_whole_dataset=False)
