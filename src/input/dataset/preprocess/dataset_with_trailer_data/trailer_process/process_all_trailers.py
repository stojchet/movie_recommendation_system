import os

from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

from src.input.dataset.preprocess.dataset_with_trailer_data.trailer_process.util.aggregation_methods import AggregationMethods
from src.input.dataset.util.util import DatasetUtil
from src.input.trailers.util.trailer_util import TrailerUtil
from src.utils.utils import TRAILER_DATA


class ProcessAllTrailers:
    def __init__(self, path: Path = TRAILER_DATA):
        self.path = path

    @staticmethod
    # TODO: should this be all movie ids?
    def __get_all_relevant_movie_ids() -> np.ndarray:
        return DatasetUtil.load_all_movie_ids()

    @staticmethod
    def get_all_movies_that_have_trailers(path: Path) -> List[str]:
        movie_ids = []
        for movie in os.listdir(path):
            if movie.startswith("."): continue
            id = movie.split("_")[0]
            movie_ids.append(id)

        return movie_ids

    @staticmethod
    def get_movies_with_no_trailer_data(path: Path = TRAILER_DATA) -> np.array:
        movie_ids: np.ndarray = ProcessAllTrailers.__get_all_relevant_movie_ids()
        intersection = np.intersect1d(movie_ids, np.array(ProcessAllTrailers.get_all_movies_that_have_trailers(path)))
        return np.setdiff1d(movie_ids, intersection).astype(np.str)

    def get_all_trailer_features(self, aggregation_type: str) -> Dict[str, tf.Tensor]:
        features_from_trailers = {}

        for movie in os.listdir(self.path):
            if movie == "movie" or movie.startswith("."): continue
            movie_id, name = movie.split("_") # what if name contains _
            keyframe_features = TrailerUtil.load_tensor(self.path.joinpath(movie, "trailer_features.npy"))
            keyframe_features = tf.squeeze(keyframe_features)
            # TODO: I dont think the aggregation is doing anything; How do I aggregate? dont create obj every time
            features_from_trailers[movie_id] = AggregationMethods().aggregate(aggregation_type, keyframe_features)

        return features_from_trailers
