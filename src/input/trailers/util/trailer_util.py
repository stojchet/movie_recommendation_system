from pathlib import Path
import re

import numpy as np
import tensorflow as tf

from src.input.dataset.util.util import DatasetUtil
from src.utils.utils import TRAILER_DATA


class TrailerUtil:
    @staticmethod
    def get_output_path(movie_id, movie_name):
        return TRAILER_DATA.joinpath(f"{movie_id}_{movie_name}")

    @staticmethod
    def get_movie_name(movie_name):
        movie_name = movie_name.split("(")[0].strip().replace(' ', '-').lower()
        return re.sub(r'[^a-zA-Z0-9-\s]', '', movie_name)

    @staticmethod
    def save_tensor(tensor: tf.Tensor, path: Path):
        DatasetUtil.save_tensor(tensor, path)

    @staticmethod
    def load_tensor(path: Path) -> tf.Tensor:
        return DatasetUtil.load_tensor(path)

    @staticmethod
    def update_ignore_file(movie_id) -> None:
        path_ignore = TRAILER_DATA.joinpath(".ignore.npy")
        elements_to_ignore = np.load(f"{path_ignore.__str__()}")
        if movie_id not in elements_to_ignore:
            elements_to_ignore = np.append(elements_to_ignore, movie_id)

            with open(f"{path_ignore.__str__()}", 'wb') as file:
                np.save(file, elements_to_ignore)
