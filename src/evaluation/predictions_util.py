import dataclasses
import pickle
from typing import List, Tuple, Union, Dict

import numpy as np
import tensorflow as tf

from input.dataset.preprocess.dataset_with_trailer_data.trailer_dataset_path_util import \
    get_path_to_trailer_dataset_tensor
from model.core.recommendation_system_core import RecommendationSystemCore
from src.input.dataset.util.util import DatasetUtil


@dataclasses.dataclass
class UserTopKRecommendations:
    k: int
    user_id: float
    recommended_movie_ids: np.array
    true_labels: np.array
    predicted_labels: np.array
    features: np.array


class ProcessPredictions:
    def __init__(self,
                 predictions: np.array,
                 config: Dict[str, Union[int, bool, str]],
                 include_trailers: bool,
                 k: int = 10):
        self.config = config
        self.__features, self.__true_labels = \
            DatasetUtil.load_X_y_tensor_datasets(include_trailers,
                                                 get_path_to_trailer_dataset_tensor(config["aggregation_method"],
                                                                                    config["feature_reduction"]))
        self.__features = self.__features.numpy()
        self.__true_labels = self.__true_labels.numpy()
        self.__movie_ids = self.__features[:, 0]
        self.__user_ids = self.__features[:, 1]

        self.__predicted_labels = predictions
        self.__k = k

    def get_k(self):
        return self.__k

    def get_true_ratings(self) -> np.array:
        return self.__true_labels

    def get_predicted_ratings(self) -> np.array:
        return self.__predicted_labels

    @staticmethod
    def get_all_movie_ids() -> np.array:
        return DatasetUtil.load_all_movie_ids()

    @staticmethod
    def save_top_k_recommendations(users_top_k: List[UserTopKRecommendations], config) -> None:
        with open(RecommendationSystemCore.get_full_path(config).joinpath("user_top_k_summaries.pkl"), 'wb') as out:
            pickle.dump(users_top_k, out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_top_k_recommendations(config) -> List[UserTopKRecommendations]:
        with open(RecommendationSystemCore.get_full_path(config).joinpath("user_top_k_summaries.pkl"), 'rb') as inp:
            users_top_k = pickle.load(inp)
        return users_top_k

    def get_top_k_recommendations(self, load=True) -> List[UserTopKRecommendations]:
        if load:
            return self.load_top_k_recommendations(self.config)

        predictions = self.__predicted_labels
        user_ids = DatasetUtil.load_all_user_ids()

        predictions_per_user: List[UserTopKRecommendations] = []
        import time
        start = time.time()
        for i, user_id in enumerate(user_ids):
            if i % 500 == 0:
                print(f"At {i}: {time.time() - start} s")
            indices = self.__get_top_recommendations_for_user(user_id, predictions, k=self.__k)
            recommended_movie_ids = self.__get_top_k_recommended_movie_ids(indices)
            if recommended_movie_ids.shape == (0,): continue
            true_labels = self.__get_top_k_recommended_true_labels(indices)
            predicted_labels = self.__get_top_k_recommended_predicted_labels(indices)
            # I think I can again just use indices from above - check if this is correct
            features = self.__get_features_for_given_movie_ids_for_user(user_id, recommended_movie_ids, indices)

            predictions_per_user.append(
                UserTopKRecommendations(k=self.__k,
                                        user_id=user_id,
                                        recommended_movie_ids=recommended_movie_ids,
                                        true_labels=true_labels,
                                        predicted_labels=predicted_labels,
                                        features=features)
            )

        self.save_top_k_recommendations(predictions_per_user, self.config)
        return predictions_per_user

    def get_list_of_topk_predictions_per_user(self, load) -> Tuple[np.array, np.array]:
        user_topk = self.get_top_k_recommendations(load=load)
        # score if type(score) != np.float64 else np.array([score])
        return (np.array([u if (u := user.true_labels).shape != () else np.array([u]) for user in user_topk], dtype=object),
                np.array([u if (u := user.predicted_labels).shape != () else np.array([u]) for user in user_topk], dtype=object))

    def __get_top_recommendations_for_user(self, user_id: float, predictions: np.ndarray, k: int = 10) -> tf.Tensor:
        # I can do this based on order of user ids ive saved in file
        # user_row_indices = tf.where(self.__user_ids == user_id)
        # if tf.size(user_row_indices) <= k:
        #     return user_row_indices
        #
        # predictions_for_user = predictions[tf.squeeze(user_row_indices).numpy()]
        # top_k_predictions_indices = tf.math.top_k(predictions_for_user, k=k).indices
        # indices_in_big_dataset = tf.gather(user_row_indices, top_k_predictions_indices)
        # return indices_in_big_dataset
        user_row_indices = np.where(self.__user_ids == user_id)[0]
        if user_row_indices.size <= k:
            return user_row_indices

        predictions_for_user = predictions[user_row_indices]
        # top_k_predictions_indices = tf.math.top_k(predictions_for_user, k=k).indices
        top_k_predictions_indices = np.argpartition(predictions_for_user, -k)[-k:]
        indices_in_big_dataset = user_row_indices[top_k_predictions_indices]
        return indices_in_big_dataset

    def __get_top_k_recommended_movie_ids(self, indices_in_big_dataset: np.array) -> np.array:
        return self.__movie_ids[indices_in_big_dataset]

    def __get_top_k_recommended_true_labels(self, indices_in_big_dataset: np.array) -> np.array:
        return np.squeeze(self.__true_labels[indices_in_big_dataset])

    def __get_top_k_recommended_predicted_labels(self, indices_in_big_dataset: np.array) -> np.array:
        return self.__predicted_labels[indices_in_big_dataset]

    def __get_features_for_given_movie_ids_for_user(self, user_id: float, top_k_movie_ids: np.array, indices_in_big_dataset: np.array) -> np.array:
        # indices_of_user = tf.reshape(tf.where(self.__user_ids == user_id), (-1,)).numpy()
        # if top_k_movie_ids.size == 1:
        #     top_k_movie_ids = np.array([top_k_movie_ids.flat[0]])
        #
        # indices_of_movies = tf.where(tf.equal(self.__movie_ids, top_k_movie_ids[:, None]))
        # # indices_of_movies = tf.reshape(indices_of_movies[:, 1], (-1,)).numpy()
        # indices_of_user_movie_tuple = tf.sets.intersection(indices_of_user[None, :], indices_of_movies[None, :]).values
        #
        # res = tf.gather(self.__features[:, 2:], indices_of_user_movie_tuple, axis=0).numpy()

        return self.__features[:, 2:][indices_in_big_dataset]
