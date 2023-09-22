import os
from typing import List, Dict, Tuple, Union

from src.input.dataset.preprocess.base_dataset.tf_tensor_dataset import TfTensorDataset
from src.input.dataset.util.util import DatasetUtil
from src.utils.constants import TRAIN, DEV, TEST
import tensorflow as tf

from src.utils.utils import BASE_TF_DATA, BASE, BASE_TENSOR_DATASET


class TfDataDataset:
    def __init__(self, process_whole_dataset: bool = False):
        self.__create_directories()
        self.__tf_datasets: Union[Dict[str, Union[Tuple[List[tf.Tensor], tf.Tensor], None]], None] = self.__get_tf_tensor_dataset(process_whole_dataset)

        if process_whole_dataset:
            self.__datasets = self.__get_multiple_input_dataset_from_tensors("whole")
            DatasetUtil.save_dataset(self.__datasets, "whole", BASE_TF_DATA)
        else:
            self.__datasets: Dict[str, tf.data.Dataset] = self.__get_final_datasets_as_tf_dataset()
            DatasetUtil.save_datasets(self.get_dict_of_datasets())

        self.__tf_datasets = None

    def get_datasets(self) -> List[tf.data.Dataset]:
        return list(self.__datasets.values())

    def get_dict_of_datasets(self) -> Dict[str, tf.data.Dataset]:
        return self.__datasets

    @staticmethod
    def __create_directories():
        if not BASE.exists(): os.mkdir(BASE)
        if not BASE_TF_DATA.exists(): os.mkdir(BASE_TF_DATA)
        if not BASE_TENSOR_DATASET.exists(): os.mkdir(BASE_TENSOR_DATASET)
        if not BASE_TENSOR_DATASET.exists(): os.mkdir(BASE_TENSOR_DATASET)

    @staticmethod
    def __get_tf_tensor_dataset(process_whole_dataset: bool) -> Dict[str, Tuple[List[tf.Tensor], tf.Tensor]]:
        tfTensorDataset: TfTensorDataset = TfTensorDataset(save_tensor_dataset=True,
                                                           process_whole_dataset=process_whole_dataset)
        return tfTensorDataset.get_dict_of_datasets()

    def __get_final_datasets_as_tf_dataset(self) -> Dict[str, tf.data.Dataset]:
        return {
            TRAIN: self.__get_multiple_input_dataset_from_tensors(TRAIN),
            DEV: self.__get_multiple_input_dataset_from_tensors(DEV),
            TEST: self.__get_multiple_input_dataset_from_tensors(TEST)
        }

    def __get_multiple_input_dataset_from_tensors(self, dataset: str) -> tf.data.Dataset:
        X, y = self.__tf_datasets[dataset]
        self.__decrease_reference_counter(dataset)
        y = tf.reshape(y, [-1])

        return tf.data.Dataset.from_tensor_slices(({"movie_ids": X[0],
                                                    "user_ids": X[1],
                                                    "movie_features": X[2],
                                                    "user_context": X[3],
                                                    }, y))

    def __decrease_reference_counter(self, dataset):
        self.__tf_datasets[dataset] = None


if __name__ == "__main__":
    import time
    start = time.time()
    TfDataDataset()
    print(time.time() - start)
