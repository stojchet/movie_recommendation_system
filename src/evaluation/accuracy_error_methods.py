# Inspo https://towardsdatascience.com/an-exhaustive-list-of-methods-to-evaluate-recommender-systems-a70c05e121de
# I think this should be calculated for different value of n for topN
from typing import Dict

from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.evaluation.predictions_util import ProcessPredictions


class AccuracyAndErrorBasedMethods:
    def __init__(self,
                 process_predictions: ProcessPredictions,
                 ):
        self.true_ratings = process_predictions.get_true_ratings()
        self.predicted_ratings = process_predictions.get_predicted_ratings()

    '''
    Input: vector of true labels and predictions
    Output: float 
    Formula: (sum((yi - yi_hat)^2)) / n
    '''
    def get_mse(self) -> float:
        return mean_squared_error(self.true_ratings, self.predicted_ratings)

    '''
    Input: vector of true labels and predictions
    Output: float 
    Formula: sqrt((sum((yi - yi_hat)^2)) / n)
    '''
    def get_rmse(self) -> float:
        return mean_squared_error(self.true_ratings, self.predicted_ratings, squared=False)

    '''
    Input: vector of true labels and predictions
    Output: float 
    Formula: (sum(|yi - yi_hat|)) / n
    '''
    def get_mae(self) -> float:
        return mean_absolute_error(self.true_ratings, self.predicted_ratings)

    def get_summary(self) -> Dict[str, float]:
        return {
            "mse": self.get_mse(),
            "rmse": self.get_rmse(),
            "mae": self.get_mae()
        }
