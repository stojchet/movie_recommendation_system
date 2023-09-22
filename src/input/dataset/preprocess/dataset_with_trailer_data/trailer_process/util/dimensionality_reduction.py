from typing import List, Callable, Dict

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# TODO: Before or after aggregation?
# TODO: add more dimensionality reduction techniques
# Note: https://towardsdatascience.com/11-dimensionality-reduction-techniques-you-should-know-in-2021-dcb9500d388b
class DimensionalityReductor:
    def __init__(self):
        self.dimensionality_reduction_methods: Dict[str, Callable] = {
            "pca": self.pca_on_trailer_features,
            "k-pca": self.kernel_pca,
            "t-sne": self.tsne,
        }

    def reduce_dimensions(self, dimensionality_reduction_method: str, features: tf.Tensor):
        return self.dimensionality_reduction_methods[dimensionality_reduction_method](features)

    @staticmethod
    def pca_on_trailer_features(features: List[tf.Tensor], n_components: int = 30) -> np.ndarray:
        normalized_features = tf.keras.utils.normalize(features)
        reshaped_features = tf.reshape(normalized_features, [normalized_features.shape[0], -1])

        pca = PCA(n_components=n_components)
        # todo: Why does this work with tensors?
        reduced_features = pca.fit_transform(reshaped_features)
        return reduced_features

    @staticmethod
    def kernel_pca(features: List[tf.Tensor]) -> np.ndarray:
        kpca = KernelPCA(n_components=30, kernel='rbf',
                         gamma=None, random_state=42)
        X_kpca = kpca.fit_transform(features)
        return X_kpca

    @staticmethod
    def tsne(features: tf.Tensor) -> np.ndarray:
        sc = StandardScaler()
        pca = PCA(n_components=30)
        tsne = TSNE()
        tsne_after_pca = Pipeline([
            ('std_scaler', sc),
            ('pca', pca),
            ('tsne', tsne)
        ])
        X_tsne = tsne_after_pca.fit_transform(features)
        return X_tsne

