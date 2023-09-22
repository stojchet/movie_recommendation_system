import numpy as np
import tensorflow_ranking as tfr
import tensorflow as tf
import scipy.stats as stats

from evaluation.predictions_util import ProcessPredictions


class RankingBasedMetrics:
    def __init__(self,
                 process_predictions: ProcessPredictions,
                 ):
        self.true_ratings, self.predicted_ratings = process_predictions.get_list_of_topk_predictions_per_user(load = False)
        self.predicted_ratings_tensor = tf.ragged.stack(list(self.predicted_ratings))
        self.get_binary_relevance_scores()
        self.get_strict_ndcg_relevance_scores()
        self.true_ratings_tensor = tf.ragged.stack(list(self.true_ratings))
        self.k = process_predictions.get_k()

    def get_binary_relevance_scores(self):
        self.binary_relevance_scores = []
        for true_ratings_user in self.true_ratings:
            score = (true_ratings_user >= 4.5) * 1.0
            self.binary_relevance_scores.append(score if type(score) != np.float64 else np.array([score]))
        self.binary_relevance_scores = tf.ragged.stack(self.binary_relevance_scores)

    def get_strict_ndcg_relevance_scores(self):
        def convert_number_to_custom_relevance(num: float):
            if num < 1.5: return 0
            if 1.5 <= num < 3.5: return 1
            if 3.5 <= num < 4: return 3
            if 4 <= num <= 4.5: return 5
            if num >= 4.5: return 10

        return self.get_ndcg(convert_number_to_custom_relevance)

    def get_looser_ndcg_relevance_scores(self):
        def convert_number_to_custom_relevance(num: float):
            if num < 1.5: return 0
            if 1.5 <= num < 3: return 1
            if 3 <= num < 4.5: return 2
            return 5

        return self.get_ndcg(convert_number_to_custom_relevance)

    def get_normal_ndcg_relevance_score(self):
        return self.get_ndcg(lambda x: x)

    def get_ndcg(self, convert_number_to_custom_relevance):
        relevance_scores = []
        for true_ratings_user in self.true_ratings:
            score = np.vectorize(convert_number_to_custom_relevance)(true_ratings_user)
            relevance_scores.append(score if type(score) != np.float64 else np.array([score]))
        relevance_scores = tf.ragged.stack(relevance_scores)
        return relevance_scores

    def calculate_mrr(self):
        mrr = tfr.keras.metrics.MRRMetric(ragged=True)
        return mrr(self.binary_relevance_scores, self.predicted_ratings_tensor).numpy()

    def calculate_ndcg(self, relevance_scores):
        ndcg = tfr.keras.metrics.NDCGMetric(ragged=True)
        return ndcg(relevance_scores, self.predicted_ratings_tensor).numpy()

    def calculate_kendall_tau(self):
        avg_ktau = []
        for true, pred in zip(self.true_ratings, self.predicted_ratings):
            ktau_one = stats.kendalltau(true, pred).statistic
            avg_ktau.append(ktau_one if ktau_one is not np.nan else 0)
        ktau = np.average(avg_ktau)
        return ktau

    def calculate_rbo(self):
        import rbo
        ratings_ = [rbo.RankingSimilarity(np.argsort(true), np.argsort(pred)).rbo() for true, pred in
                    zip(self.true_ratings, self.predicted_ratings)]
        rbo = np.average(ratings_)
        return rbo

    def calculate_map(self):
        map = tfr.keras.metrics.MeanAveragePrecisionMetric(topn=self.k)
        return map(self.binary_relevance_scores, self.predicted_ratings_tensor).numpy()


    def get_metrics_summary(self):
        return {
            "mrr": self.calculate_mrr(),
            "map": self.calculate_map(),
            "strict-ndcg": self.calculate_ndcg(self.get_strict_ndcg_relevance_scores()),
            "looser-ndcg": self.calculate_ndcg(self.get_looser_ndcg_relevance_scores()),
            "normal-ndcg": self.calculate_ndcg(self.get_normal_ndcg_relevance_score()),
            "kendall-tau": self.calculate_kendall_tau(),
            "rbo": self.calculate_rbo(),
        }
