from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import sklearn

from src.input.dataset.preprocess.movielens_preprocess.dataset_preprocess import Preprocess
from src.input.dataset.util.util import DatasetUtil
from src.utils.constants import TRAIN, DEV, TEST


class PdDataFrameDataset:
    def __init__(self, process_whole_dataset: bool):
        self.__user_context_columns: List[str] = ["weekend", "daytime"]
        self.__get_X_and_y()
        self.__get_config()
        DatasetUtil.save_all_movie_and_user_ids(self.X)

        if process_whole_dataset:
            self.__datasets = {
                "whole": (self.__prepare_input_features(self.X, self.__user_context_columns),
                         self.y)
            }
        else:
            self.__datasets: Dict[str, Tuple[List[pd.DataFrame], pd.DataFrame]] = self.__get_final_datasets()

    def get_datasets(self) -> List[pd.DataFrame]:
        return list(sum([self.__datasets[TRAIN], self.__datasets[DEV], self.__datasets[TEST]], ()))

    def get_dict_of_datasets(self) -> Dict[str, Tuple[List[pd.DataFrame], pd.DataFrame]]:
        return self.__datasets

    def __get_X_and_y(self):
        preprocessed = Preprocess()
        self.X = preprocessed.X
        self.y = preprocessed.y

    def __get_config(self):
        max_movie_id = int(self.X.loc[self.X.movieId.idxmax()].movieId) + 1
        max_user_id = int(self.X.loc[self.X.userId.idxmax()].userId) + 1

        # This is shape
        movie_features_shape = int(
            np.subtract(self.X.shape[1:], np.array([len(self.__user_context_columns) + 2]))[0])  # 24
        user_features_shape = int(np.array([len(self.__user_context_columns)])[0])  # 2
        movie_id_shape = self.X.movieId.shape[1:]
        user_id_shape = self.X.userId.shape[1:]

        dict = {
            "max_movie_id": max_movie_id,
            "max_user_id": max_user_id,
            "movie_features_shape": movie_features_shape,
            "user_features_shape": user_features_shape,
            "movie_id_shape": movie_id_shape,
            "user_id_shape": user_id_shape,
        }

        DatasetUtil.save_config(dict)

    def __get_final_datasets(self) -> Dict[str, Tuple[List[pd.DataFrame], pd.DataFrame]]:
        train_X, train_y, dev_X, dev_y, test_X, test_y = self.__split_datasets()
        train_X: List[pd.DataFrame] = self.__prepare_input_features(train_X, self.__user_context_columns)
        dev_X: List[pd.DataFrame] = self.__prepare_input_features(dev_X, self.__user_context_columns)
        test_X: List[pd.DataFrame] = self.__prepare_input_features(test_X, self.__user_context_columns)

        return {
            TRAIN: (train_X, train_y),
            DEV: (dev_X, dev_y),
            TEST: (test_X, test_y)
        }

    def __split_datasets(self) \
            -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_X, X_rest, train_y, y_rest = sklearn.model_selection.train_test_split(self.X,
                                                                                    self.y,
                                                                                    train_size=0.7)

        self.__decrease_reference_counter()

        dev_X, test_X, dev_y, test_y = sklearn.model_selection.train_test_split(X_rest,
                                                                                y_rest,
                                                                                train_size=0.1)

        return DatasetUtil.create_df(train_X), \
            DatasetUtil.create_df(train_y), \
            DatasetUtil.create_df(dev_X), \
            DatasetUtil.create_df(dev_y), \
            DatasetUtil.create_df(test_X), \
            DatasetUtil.create_df(test_y)

    def __decrease_reference_counter(self):
        self.X = None
        self.y = None

    @staticmethod
    def __prepare_input_features(dataset: pd.DataFrame, user_context_columns: List[str]) -> List[pd.DataFrame]:
        movie_ids = dataset.movieId
        user_ids = dataset.userId
        dataset = dataset.drop(["movieId", "userId"], axis=1)

        user_context = dataset[user_context_columns]
        dataset = dataset.drop(user_context_columns, axis=1)

        final_dataset = [movie_ids, user_ids, dataset, user_context]

        return final_dataset


if __name__ == "__main__":
    PdDataFrameDataset(False)
