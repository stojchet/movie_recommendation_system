import math
from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.evaluation.predictions_util import ProcessPredictions, UserTopKRecommendations


class GeneralPerformanceMetrics:
    def __init__(self, predictions_process: ProcessPredictions):
        self.top_k_recommendations_movie_ids: List[UserTopKRecommendations] \
            = predictions_process.get_top_k_recommendations()
        self.predictions_process = predictions_process

    def call_coverage(self):
        return self.coverage(self.predictions_process.get_all_movie_ids(), self.top_k_recommendations_movie_ids)

    '''
    Input: list of UserTopKRecommendations
    Output: float 
    Formula: % of movies that were ever recommended to any user
     set(recommended_movies).size / all_movies.size
    '''
    @staticmethod
    def coverage(all_movie_ids: np.array, top_k_recommendations_movie_ids: List[UserTopKRecommendations]):
        covered_movies = set()
        for user_top_k_recommendations in top_k_recommendations_movie_ids:
            recommendations = user_top_k_recommendations.recommended_movie_ids
            if recommendations.size == 1:
                recommendations = np.empty(shape=[])

            recommendations = set(tuple(np.intersect1d(recommendations, all_movie_ids)))
            covered_movies = covered_movies.union(recommendations)

        return len(covered_movies) / len(all_movie_ids)

    '''
    Input: list of UserTopKRecommendations
    Output: float 
    Formula: 1 - cosine similarity 
    '''
    @staticmethod
    def personalization(top_k_recommendations_movie_ids: List[UserTopKRecommendations]):
        v = [np.array(e.recommended_movie_ids) for e in top_k_recommendations_movie_ids]
        # Should I always be able to get 10 recommendations?
        # Not necessarily - if in the test dataset I have user_id = 1 with 1 movie then I will get 1 solution
        matrix_of_movie_ids = np.vstack(list(filter(lambda x: x.size >= 10, list(v))))
        similarities = cosine_similarity(matrix_of_movie_ids)
        total_similarity = np.sum(np.triu(similarities)) / GeneralPerformanceMetrics.get_upper_triu_size(
            similarities.size)
        return 1 - total_similarity

    @staticmethod
    def get_upper_triu_size(n):
        return math.sqrt(n) * (math.sqrt(n) + 1) / 2

    '''
    Input: list of UserTopKRecommendations
    Output: float 
    Formula: sum(sum(sim(i, j))) -> calculated average of sum of similarity matrix (of 2 user lists)
    '''
    @staticmethod
    def intra_list_similarity(top_k_recommendations_movie_ids: List[UserTopKRecommendations]):
        # https://towardsdatascience.com/evaluation-metrics-for-recommender-systems-df56c6611093
        sum_cosine_similarity = 0
        for user_top_k_recommendations in top_k_recommendations_movie_ids:
            similarities = cosine_similarity(user_top_k_recommendations.features)
            total_similarity = np.sum(np.triu(similarities)) / GeneralPerformanceMetrics.get_upper_triu_size(similarities.size)
            sum_cosine_similarity += total_similarity

        return sum_cosine_similarity / len(top_k_recommendations_movie_ids)

    def get_metrics_summary(self):
        return {
            # "personalization": self.personalization(self.top_k_recommendations_movie_ids),
            "coverage": self.call_coverage(),
            "intra_list_similarity": self.intra_list_similarity(self.top_k_recommendations_movie_ids)
        }