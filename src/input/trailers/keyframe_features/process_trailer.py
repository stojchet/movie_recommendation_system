import os
from pathlib import Path
from typing import List

from src.input.trailers.keyframe_features.process_image_model import ImageFeatureExtractor
from src.input.trailers.util.trailer_util import TrailerUtil

from src.utils.utils import TRAILER_DATA
import tensorflow as tf


class ProcessTrailer:
    def __init__(self, path: Path = TRAILER_DATA, model_type: str = "vgg"):
        self.path = path
        self.model = ImageFeatureExtractor(model_type)

    def download_helper(self, movieId, title):
        return self.__get_tensor_matrix(self.__get_list_of_features_for_all_keyframes(movieId, title))

    def __get_list_of_features_for_all_keyframes(self, movieId, title) -> List[tf.Tensor]:
        all_keyframes, movie_path = self.__get_all_keyframes_for_movie(movieId, title)
        all_features = []
        for keyframe in all_keyframes:
            features = self.__process_image(movie_path.joinpath(keyframe))
            all_features.append(features)
        return all_features

    @staticmethod
    def __get_tensor_matrix(features: List[tf.Tensor]):
        return tf.stack(features)

    @staticmethod
    def __get_all_keyframes_for_movie(movieId, title):
        movie_name = TrailerUtil.get_movie_name(title)
        movie_path = TrailerUtil.get_output_path(movieId, movie_name).joinpath("keyframes")
        all_keyframes = os.listdir(movie_path)
        return all_keyframes, movie_path

    def __process_image(self, image_path) -> tf.Tensor:
        return self.model.extract_features(image_path)
