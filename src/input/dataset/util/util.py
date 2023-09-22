import json
import os
from json import JSONDecodeError
from pathlib import Path
from typing import Tuple, List, Dict, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils.constants import TRAIN, DEV, TEST, PREDICTIONS
from src.utils.utils import BASE_TF_DATA, DATA_DIR, BASE_TENSOR_DATASET, WITH_TRAILER_TENSOR_DATASET


class DatasetUtil:
    # Util
    @staticmethod
    def create_df(X: np.array) -> pd.DataFrame:
        return pd.DataFrame(X)

    @staticmethod
    def load_X_y_tensor_datasets(include_trailer: bool, path: Path = None) -> Tuple[tf.Tensor, tf.Tensor]:
        if path == None:
            path = WITH_TRAILER_TENSOR_DATASET if include_trailer else BASE_TENSOR_DATASET
        return DatasetUtil.load_tensor_dataset(TEST, path)

    @staticmethod
    def load_predictions(include_trailer: bool):
        path = WITH_TRAILER_TENSOR_DATASET if include_trailer else BASE_TENSOR_DATASET
        return DatasetUtil.load_numpy_array(path.joinpath(PREDICTIONS))

    # Load / Save Datasets
    @staticmethod
    def save_datasets(datasets: Dict[str, Union[tf.data.Dataset, tf.Tensor]], path: Path = BASE_TF_DATA):
        DatasetUtil.save_dataset(datasets[TRAIN], TRAIN, path)
        DatasetUtil.save_dataset(datasets[DEV], DEV, path)
        DatasetUtil.save_dataset(datasets[TEST], TEST, path)

    @staticmethod
    def save_dataset(dataset: Union[tf.data.Dataset, tf.Tensor], dataset_type: str, path: Path):
        dataset.save(path.joinpath(dataset_type).__str__())

    @staticmethod
    def load_dataset(dataset_type: str, path: Path):
        return tf.data.Dataset.load(path.joinpath(dataset_type).__str__())

    @staticmethod
    def save_all_tensor_datasets(datasets: Dict[str, Tuple[tf.Tensor, tf.Tensor]]):
        for dataset_type, dataset in datasets.items():
            features, labels = dataset
            DatasetUtil.save_tensor_dataset_of_type(dataset_type, features, labels)

    @staticmethod
    def save_tensor_dataset_of_type(dataset_type, features, labels, path: Path = BASE_TENSOR_DATASET):
        DatasetUtil.save_tensor_dataset_features(features, dataset_type, path)
        DatasetUtil.save_tensor_dataset_labels(labels, dataset_type, path)

    @staticmethod
    def save_tensor_dataset_features(features: tf.Tensor, dataset_type: str, path: Path):
        if not os.path.isdir(path.joinpath(dataset_type)):
            os.mkdir(path.joinpath(dataset_type))

        tensor_features = tf.concat(features, axis=1)
        DatasetUtil.save_tensor(tensor_features, path.joinpath(dataset_type, "features"))

    @staticmethod
    def save_tensor_dataset_labels(labels: tf.Tensor, dataset_type: str, path: Path):
        DatasetUtil.save_tensor(labels, path.joinpath(dataset_type, "labels"))

    @staticmethod
    def load_tensor_dataset(dataset_type: str, path: Path) -> Tuple[tf.Tensor, tf.Tensor]:
        return DatasetUtil.load_tensor_dataset_features(dataset_type, path), \
            DatasetUtil.load_tensor_dataset_labels(dataset_type, path)

    @staticmethod
    def load_tensor_dataset_features(dataset_type: str, path: Path) -> tf.Tensor:
        return DatasetUtil.load_tensor(path.joinpath(dataset_type, "features"))

    @staticmethod
    def load_tensor_dataset_labels(dataset_type: str, path: Path) -> tf.Tensor:
        return DatasetUtil.load_tensor(path.joinpath(dataset_type, "labels"))

    @staticmethod
    def save_tensor(tensor: tf.Tensor, path: Path):
        with open(f"{path.__str__()}", 'wb') as file:
            np.save(file, tensor.numpy())

    @staticmethod
    def load_tensor(path: Path) -> tf.Tensor:
        array = np.load(f"{path.__str__()}")
        return tf.convert_to_tensor(array)

    @staticmethod
    def save_numpy_array(arr: np.array, path: Path):
        with open(f"{path.__str__()}.npy", 'wb') as file:
            np.save(file, arr)

    @staticmethod
    def load_numpy_array(path: Path) -> np.array:
        return np.load(f"{path.__str__()}.npy")

    @staticmethod
    def save_dict(dict: Dict, path: Path):
        if not path.exists():
            open(path, "x").close()
        with open(path, 'r+') as config_file:
            try:
                old_config = json.loads(config_file.read())
                config_file.seek(0)
                config_file.truncate()
            except JSONDecodeError:
                old_config = {}
            old_config.update(dict)
            json.dump(old_config, config_file)

    @staticmethod
    def load_dict(path: Path) -> Dict[str, Union[int, List]]:
        with open(path, 'rb') as config_file:
            return json.loads(config_file.read())

    @staticmethod
    def save_config(config: Dict):
        DatasetUtil.save_dict(config, DATA_DIR.joinpath("config.json"))

    @staticmethod
    def load_config() -> Dict[str, Union[int, List]]:
        return DatasetUtil.load_dict(DATA_DIR.joinpath("config.json"))

    @staticmethod
    def save_all_movie_and_user_ids(X: pd.DataFrame):
        with open(DATA_DIR.joinpath("all_movie_ids.npy"), "wb") as file:
            np.save(file, np.unique(X.movieId.astype(dtype=float)))

        with open(DATA_DIR.joinpath("all_user_ids.npy"), "wb") as file:
            np.save(file, np.unique(X.userId.astype(dtype=float)))

    @staticmethod
    def load_all_movie_ids():
        with open(DATA_DIR.joinpath("all_movie_ids.npy"), "rb") as file:
            movie_ids: np.ndarray = np.load(file)
        return movie_ids

    @staticmethod
    def load_all_user_ids():
        with open(DATA_DIR.joinpath("all_user_ids.npy"), "rb") as file:
            return np.load(file)
