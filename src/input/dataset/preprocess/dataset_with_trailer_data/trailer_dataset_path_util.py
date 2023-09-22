from utils.utils import WITH_TRAILER


BASE_PATH = WITH_TRAILER


def get_path_to_trailer_dataset_tensor(aggregation_method: str, dimensionality_reduction_method):
    return get_base_trailer_dataset_path(aggregation_method, dimensionality_reduction_method).joinpath("tensor")


def get_path_to_trailer_dataset_tf_data(aggregation_method: str, dimensionality_reduction_method):
    return get_base_trailer_dataset_path(aggregation_method, dimensionality_reduction_method).joinpath("tf_data")


def get_base_trailer_dataset_path(aggregation_method: str, dimensionality_reduction_method):
    return BASE_PATH.joinpath(f"{aggregation_method}_{dimensionality_reduction_method}")

