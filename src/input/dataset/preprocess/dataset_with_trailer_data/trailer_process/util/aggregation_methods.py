import tensorflow as tf


class AggregationMethods:
    def __init__(self):
        self.__aggregation_methods = {
            "max": self.__max_pooling,
            "avg": self.__average_pooling,
        }

    def aggregate(self, aggregation_type: str, features: tf.Tensor):
        return self.__aggregation_methods[aggregation_type](features)

    @staticmethod
    def __max_pooling(features: tf.Tensor) -> tf.Tensor:
        pooled_features = tf.math.reduce_max(features, axis=0)
        return pooled_features

    @staticmethod
    def __average_pooling(features: tf.Tensor) -> tf.Tensor:
        pooled_features = tf.math.reduce_mean(features, axis=0)
        return pooled_features
