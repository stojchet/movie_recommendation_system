from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
import keras.utils as image
import numpy as np
import keras
import tensorflow as tf


class ImageFeatureExtractor:
    def __init__(self,
                 model_type: str = "vgg",
                 *args,
                 **kwargs):
        if model_type == 'resnet':
            self.model = ResNet50(weights='imagenet',
                                  include_top=False,
                                  pooling='avg')
        elif model_type == 'vgg':
            self.model = VGG16(weights='imagenet',
                               include_top=False,
                               pooling='avg')

        self.model_type = model_type

    def extract_features(self, img_path):

        x = image.load_img(img_path, color_mode='rgb', target_size=(224, 224))
        x = image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        if self.model_type == "resnet":
            x = keras.applications.resnet.preprocess_input(x)

        elif self.model_type == "vgg":
            x = keras.applications.vgg16.preprocess_input(x)

        features = self.model.predict(x)
        eager_tensor = tf.constant(np.array(features), dtype=tf.float32)

        return tf.squeeze(eager_tensor)

# ImageFeatureExtractor().extract_features("/home/teodora/Personal/Uni/isp_project/movie_recommendation_system/trailer_data/1.visualization.png")