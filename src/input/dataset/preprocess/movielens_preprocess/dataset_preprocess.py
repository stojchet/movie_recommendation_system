import re
from datetime import datetime

import pandas as pd
import numpy as np

import sklearn.model_selection
from sklearn.preprocessing import *

from src.utils.utils import MOVIELENS_MOVIES_PATH, MOVIELENS_RATINGS_PATH
from src.input.dataset.preprocess.util.movielens_constants import *


class Preprocess:
    def __init__(self,
                 seed: int = 42):
        self.__movies_df: pd.DataFrame = pd.read_csv(MOVIELENS_MOVIES_PATH)
        self.__ratings_df: pd.DataFrame = pd.read_csv(MOVIELENS_RATINGS_PATH)
        self.__merged_dataset = None

        self.random: np.random.RandomState = np.random.RandomState(seed)
        self.__preprocess()
        self.__decrease_reference_counter()

    def __decrease_reference_counter(self):
        self.__movies_df = None
        self.__ratings_df = None
        self.__merged_dataset = None

    def __get_unique_genres(self):
        return np.unique([j for i in self.__movies_df.genres.str.split('|') for j in i])

    def __genres_to_multi_label_encoding(self):
        all_genres = self.__get_unique_genres()
        self.__movies_df.genres = self.__movies_df.genres.str.split("|")
        encoded_genres = pd.DataFrame(MultiLabelBinarizer(classes=all_genres).fit_transform(self.__movies_df.genres),
                                      columns=all_genres)

        self.__movies_df = self.__movies_df.drop(GENRES, axis=1)
        self.__movies_df = pd.concat([self.__movies_df, encoded_genres], axis=1, sort=False)

    def __add_mean_movie_rating(self):
        average_ratings = self.__ratings_df.groupby(MOVIE_ID)[RATING].mean()
        number_ratings = self.__ratings_df.groupby(MOVIE_ID)[RATING].count()
        average_movie_ratings = average_ratings.to_frame().join(number_ratings, on=MOVIE_ID, lsuffix="_average",
                                                                rsuffix="_count")

        self.__movies_df = self.__movies_df.join(average_movie_ratings, on=MOVIE_ID)
        self.__movies_df.rating_average = sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(
            self.__movies_df.rating_average.values.reshape(-1, 1))[:, 0]

    @staticmethod
    def __extract_release_year(title_col) -> int:
        try:
            return int(title_col.split("(")[-1].replace(")", "").strip())
        except Exception:
            return np.nan

    def __split_tite_and_year(self):
        self.__movies_df[DATE] = self.__movies_df[TITLE].apply(self.__extract_release_year)
        self.__movies_df[TITLE] = self.__movies_df[TITLE].apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x).strip())
        self.__movies_df[DATE] = self.__movies_df[DATE].fillna(9999)
        self.__movies_df["old"] = self.__movies_df[DATE].apply(lambda x: 1 if x < 2000 else 0)

    def __normalize_user_ratings(self):
        self.__ratings_df.rating = sklearn.preprocessing \
                                     .MinMaxScaler(feature_range=(0, 1)) \
                                     .fit_transform(self.__ratings_df.rating.values.reshape(-1, 1))[:, 0]

    def __convert_timestep_to_correct_format(self):
        self.__ratings_df[TIMESTAMP] = self.__ratings_df[TIMESTAMP].apply(lambda x: datetime.fromtimestamp(x))
        self.__ratings_df[DAYTIME] = self.__ratings_df[TIMESTAMP].apply(lambda x: 1 if 6 < int(x.strftime("%H")) < 20 else 0)
        self.__ratings_df[WEEKEND] = self.__ratings_df[TIMESTAMP].apply(lambda x: 1 if x.weekday() in [5, 6] else 0)
        self.__ratings_df = self.__ratings_df.drop(TIMESTAMP, axis=1)

    def __process_movie_df(self):
        self.__genres_to_multi_label_encoding()
        self.__add_mean_movie_rating()
        self.__split_tite_and_year()

    def __process_user_df(self):
        # self._normalize_user_ratings()
        self.__convert_timestep_to_correct_format()

    def __remove_redundant_rows(self):
        self.__ratings_df = self.__ratings_df.dropna(how="any", axis=0)
        self.__movies_df = self.__movies_df.dropna(how="any", axis=0)

    def __prepare_merged_dataset(self):
        self.__merged_dataset = pd.merge(self.__ratings_df, self.__movies_df, on=MOVIE_ID).sort_values(USER_ID)
        self.__merged_dataset = self.__merged_dataset.sample(axis=0, frac=1, random_state=self.random)
        self.__merged_dataset = self.__merged_dataset.drop(TITLE, axis=1)

    def __split_examples_and_labels(self):
        self.y = self.__merged_dataset[RATING]
        self.X = self.__merged_dataset.drop(RATING, axis=1)

    def __preprocess(self):
        self.__process_movie_df()
        # There's a bottleneck here
        self.__process_user_df()
        self.__remove_redundant_rows()
        # Fails here
        self.__prepare_merged_dataset()
        self.__split_examples_and_labels()
