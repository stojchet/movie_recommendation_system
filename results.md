# Documentation for Results and Reproducibility:

## Introduction:
Welcome to the documentation for reproducing the results of the recommendation system designed to provide recommendations to users based on their past interactions, which is enriched by adding features extracted from the trailer of the movies. This document is intended for readers who are curious about the outcomes of our recommendation system and would like to replicate the results using their own environment. By following the steps outlined here, you will be able to recreate the experiments and evaluate the system's performance metrics as presented in our study.

Within this documentation, we provide a detailed guide to setting up the required environment, accessing the necessary dataset, preprocessing the data, training the recommendation model, and evaluating its performance. By adhering to these steps and configurations, you should achieve comparable results to those reported in our study. Additionally, we offer troubleshooting advice to address common issues that might arise during the reproduction process, ensuring a smooth and accurate replication of our recommendation system's outcomes.

## Environment Setup:
All dependencies are listed in the requirements.txt file. 
Please run 

```shell
pip install requirements.txt
```

## Data and Model Availability:
I am using the movielens dataset. To get the dataset I am training the model on, please run
```shell
src/input/dataset/preprocess/base_dataset/tf_data_dataset.py
```

This will create the base dataset. 
To get the trailer data please run 
```shell
src/input/trailers/keyframe_features/all_features.py
```
Now you have two separate datasets you need to merge. 
Please run 
```shell
src/input/dataset/preprocess/dataset_with_trailer_data/dataset_with_trailer_data.py
```
Note: you can change the aggregation type and dimensionality reduction method if you'd like. 
You can run:
```shell
python src/input/dataset/preprocess/dataset_with_trailer_data/dataset_with_trailer_data.py --aggregation="max" --dimensionality_reduction="pca"
```
Possible aggregation methods are `max` and `avg`.
Possible dimensionality reduction methods are `pca`, `k-pca`, `t-sne`
This will create the trailer enriched dataset.
To get the filtered dataset, please run
```shell
src/input/dataset/preprocess/base_dataset/filtered_dataset_to_match_trailer_dataset.py
```

Note: dimensions of the base dataset, regardless if it's the whole dataset or filtered will be (None, 28)
The size of the whole dataset is: 25_000_095
The size of the filtered dataset depends on your YouTube permissions as well as the parameters you set, for example max video duration you will process. 
If you follow this run you should get: 12_027_582
Here are some stats for the dataset:
```{"max_movie_id": 209172, "max_user_id": 162542, "movie_features_shape": 24, "user_features_shape": 2, "movie_id_shape": [], "user_id_shape": []}```
As you can see I've split the 28 mentioned features in 4 parts: movie_features_shape, user_features_shape, movie_id_shape, user_id_shape.
Dimensions of the trailer enriched dataset will be 28 + embedding_size, which in my case is 30, so final dataset size is (None, 54)
You can view dataset heads in `dataset_examination.ipynb`

## Expected Results:
### Trailer based model
The best model I trained on trailer data was with maximum aggregation and PCA feature reduction method. The results I got are:
{"mse": "0.9741376", "rmse": "0.9869841", "mae": "0.7613914"}
{"coverage": "0.016766304808034278", "intra_list_similarity": "0.9950840327605363"}
{"mrr": "0.24160278", "strict-ndcg": "0.9547026", "looser-ndcg": "0.95654404", "normal-ndcg": "0.9962901", "kendall-tau": "0.02173382595958153", "rbo": "0.9838282422619772"}
The most relevant score is MSE that is also the loss on which I am training the models.

The model trained on base dataset without trailer features performed very poorly compared to the one above. The metrics are as follows:
{"mse": "1.6590896", "rmse": "1.2880565", "mae": "0.9834954"}
{"coverage": "0.016766304808034278", "intra_list_similarity": "0.9950840322310077"}
{"mrr": "0.24027422", "strict-ndcg": "0.9516269", "looser-ndcg": "0.95443106", "normal-ndcg": "0.9941391", "kendall-tau": "0.000806241656154251", "rbo": "0.9784146687761145"}

Actually all the models trained on data including trailer features (even the worst model) perform better than the base model. See details in `metrics` folder.

## Acknowledgments:
I used [TransNetV2](https://arxiv.org/abs/2008.04838) to extract keyframes from the trailers. However, the output of the model need to be further processed for which I used the viretpipeline.
