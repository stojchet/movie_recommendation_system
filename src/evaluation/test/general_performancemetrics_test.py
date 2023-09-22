from evaluation.general_preformance_methods import GeneralPerformanceMetrics
from evaluation.predictions_util import UserTopKRecommendations
import numpy as np

EPSILON = 1e-3


def test_personalization():
    user_top_k_Recommendations_list = [
        UserTopKRecommendations(10, 1.0, np.array([1, 2, 3]), np.empty([])),
        UserTopKRecommendations(10, 2.0, np.array([1, 2, 3]), np.empty([])),
    ]
    assert GeneralPerformanceMetrics.personalization(user_top_k_Recommendations_list) == 0


# Below tests don't work, but I'm not sure if they should
def test_personalization_1():
    user_top_k_Recommendations_list = [
        UserTopKRecommendations(10, 1.0, np.array([1, 2, 3, 4]), np.empty([])),
        UserTopKRecommendations(10, 2.0, np.array([1, 2, 5, 6]), np.empty([])),
    ]
    assert GeneralPerformanceMetrics.personalization(user_top_k_Recommendations_list) == 0.5


def test_personalization_2():
    user_top_k_Recommendations_list = [
        UserTopKRecommendations(10, 1.0, np.array([1, 2, 3, 4]), np.empty([])),
        UserTopKRecommendations(10, 2.0, np.array([1, 2, 5, 6]), np.empty([])),
        UserTopKRecommendations(10, 2.0, np.array([1, 2, 3, 6]), np.empty([])),
    ]
    # Similarities = 0.5, 0.75, 0.75
    assert GeneralPerformanceMetrics.personalization(user_top_k_Recommendations_list) == 2 / 3


def test_coverage():
    # Note: movie ids come from train set
    all_movie_ids = [1, 2, 3, 4]
    user_top_k_Recommendations_list = [
        UserTopKRecommendations(10, 1.0, np.array([1, 1, 1]), np.empty([])),
        UserTopKRecommendations(10, 2.0, np.array([1, 1, 1]), np.empty([])),
        UserTopKRecommendations(10, 3.0, np.array([1, 1, 1]), np.empty([])),
    ]

    assert GeneralPerformanceMetrics.coverage(all_movie_ids, user_top_k_Recommendations_list) == 0.25


def test_intra_list_similarity():
    user_top_k_Recommendations_list = [
        UserTopKRecommendations(10, 1.0, np.array([1, 1, 1]), np.array([[1, 2, 3, 4], [1, 2, 3, 4]])),
    ]

    assert abs(GeneralPerformanceMetrics.intra_list_similarity(user_top_k_Recommendations_list) - 1) < EPSILON
