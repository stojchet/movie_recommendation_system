import tensorflow as tf

from input.dataset.preprocess.dataset_with_trailer_data.trailer_dataset_path_util import get_path_to_trailer_dataset_tf_data
from src.input.dataset.util.util import DatasetUtil
from src.utils.utils import BASE_TF_DATA, WITH_TRAILER_TF_DATA, BASE_FILTERED


class Dataset:
    def __init__(self,
                 dataset_type: str,
                 include_trailers: bool,
                 aggregation_method: str,
                 dimensionality_reduction: str,
                 filtered_dataset: str = False):
        self.dataset_type = dataset_type
        self.include_trailers = include_trailers
        self.aggregation_method = aggregation_method
        self.dimensionality_reduction = dimensionality_reduction
        self.filtered_dataset = filtered_dataset

    def get_dataset(self) -> tf.data.Dataset:
        if self.include_trailers:
            return DatasetUtil.load_dataset(
                self.dataset_type,
                get_path_to_trailer_dataset_tf_data(
                    aggregation_method=self.aggregation_method,
                    dimensionality_reduction_method=self.dimensionality_reduction))
        else:
            if self.filtered_dataset:
                return DatasetUtil.load_dataset(self.dataset_type, BASE_FILTERED)
            return DatasetUtil.load_dataset(self.dataset_type, BASE_TF_DATA)
