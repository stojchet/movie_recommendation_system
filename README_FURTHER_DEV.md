# Documentation for Further Development:

## Introduction:
Welcome to the documentation for the recommendation system designed to provide recommendations to users based on their past interactions, which is enriched by adding features extracted from the trailer of the movies. This documentation is aimed at developers, data scientists, and researchers who are interested in understanding the inner workings of the recommendation system and exploring avenues for further development and enhancement.

In this documentation, you will find a comprehensive overview of the architecture, algorithms, and methodologies that power the recommendation system. I will delve into the data preprocessing steps, model training techniques, and evaluation metrics used to craft accurate and effective recommendations. Whether you're looking to optimize the current system, experiment with different algorithms, or integrate new features, this documentation will serve as your guide to navigating the intricacies of our recommendation system's development.

## Architecture and Model Overview:
Model is stored in `src/model`. 
There are 2 input streams to the model the base dataset and the trailer features which are being concatenated by `src/model/dataset/merge_datasets.py`.
The body of model mainly consists of 3 preactivation layers that contain in the following order:
- Batch normalization
- Dense layer
- Activation (leaky ReLu)
- Dropout
At the end there is a Lambda layer that modifies the output to be in the right interval - [1, 5].
To get a better idea of the architecture please look at [add location]

## Data Preprocessing:
Detail the data sources used for training and testing the model.
Describe the data preprocessing steps, including data cleaning, handling missing values, and feature engineering.
Provide information about how you split the data into training, validation, and testing sets.

### Dataset preprocessing
I am using the MovieLens dataset. Specifically the ml-25m version.
From the dataset I am utilizing the ratings.csv and movies.csv datasets, which are being joined on movieId. 
Then there are a few steps for filtering the dataset:
* Extracting and encoding unique movie genres using multi-label binary encoding.
* Calculating and incorporating mean movie ratings and the number of ratings per movie.
* Extracting release years from movie titles and removing year information from titles.
* Normalizing user ratings and converting Unix timestamps to human-readable format.
* Creating binary features for daytime and weekend interactions based on timestamps.
* Cleaning the data by removing rows with missing values.
* Separating the target variable (ratings) from the features for further use.
You can find the script that cleans up the dataset in `src/input/dataset/preprocess/movielens_preprocess/dataset_preprocess.py`. This loads the csv files as pandas dataframes and continues to work with pandas dataframes.
Then there are 2 other scripts that save the dataset in 2 types: tensor and tf.data.Dataset
* `src/input/dataset/preprocess/base_dataset/tf_tensor_dataset.py`
* `src/input/dataset/preprocess/base_dataset/tf_data_dataset.py`
In general the model mostly works with the tf.data.Dataset.
To get the dataset please run 
 ```shell
python src/input/dataset/preprocess/base_dataset/tf_data_dataset.py
```
You just need to have the MovieLens dataset saved in `ml-25m`, specifically the files `ratings.csv` and `movies.csv`. The script will create and save the tensor and tf.data.Dataset version of the final dataset.

### Tailer feature extraction
Now I am utilizing the extended version of the MovieLens dataset that contains youtubeId movieId pairs. 
For each trailer I do 3 main steps:

1. Download trailer
I assess whether it fulfills specific conditions before proceeding. These conditions ensure that the trailer hasn't been processed before and isn't included in an "ignore" list in which I collect all trailers that for some reason failed to be processed. If the criteria are met, I download the trailer and evaluate its duration, discarding trailers that exceed five minutes.
2. Extract features
Upon verification, I extract keyframes using TransNetV2. Subsequently, I employ the ImageFeatureExtractor class that uses [resnet/vgg] model to extract features and save them. 
3. Delete trailer
No need to save redundant data

Essentially, this workflow automates the complete pipeline for trailer processing, keyframe extraction, feature acquisition, and storage. 
These features will be stored locally, after that you will create the datasets and disregard these features.
To collect trailer features please run 
```shell
python src/input/trailers/keyframe_features/all_features.py
```
This will save all trailer features for all trailers it can collect features from. Trailers will be stored in `trailer_data`, specifically each movie will be stored in `trailer_data/{movie_id}_{movie_name}`

### Trailer enriched dataset 
In my extended data augmentation approach, I'm taking the foundational MovieLens dataset and infusing it with valuable visual insights extracted from movie trailers, using the process I described above.

To get the final dataset, two distinct aggregation techniques—average and maximum—are employed to distill these features across all keyframes within each movie. Furthermore, I explore dimensionality reduction via three strategies: Principal Component Analysis (PCA), Kernel PCA (K-PCA), and t-Distributed Stochastic Neighbor Embedding (t-SNE), which enable the condensation of information while preserving relevant patterns. This amalgamated dataset fuses the original attributes, user ratings, and the extracted trailer-based features. Through this effort, I aspire to unlock the potential for delivering more nuanced and contextually sensitive movie recommendations to users, elevating the overall recommendation experience.
To get the dataset please run
```shell
python src/input/dataset/preprocess/dataset_with_trailer_data/dataset_with_trailer_data.py
```

### Filtered dataset
Because not all movies contain trailers that can be downloaded (trailer that can be downloaded are determined based on the heuristics I've defined in `Download trailer` section), the trailer infused dataset doesn't contain all datapoints.
To remove this possible bias I create a filtered dataset, where I remove all movie user pairs for which I wasn't able to extract features from the trailer. I use this to train the base model.
You can run the following to get the filtered dataset:
```shell
python src/input/dataset/preprocess/base_dataset/filtered_dataset_to_match_trailer_dataset.py
```

## Model Training:
Model is stored in `src/model`.
The dataset the model trains on depends on a few things. First of all whether or not to include trialers. If no then we train on the base dataset, if yes, we train on the dataset with trailers, that can also be sorted by aggregation methods (max, min) and dimensionality reduction methods (pca, k-pca, t-sne). So for example we might train on trailer enriched dataset with whose keyframe features were aggregated by max and then reduced by pca.
I have a basic unit called Preactivation Layer that consists of a batch normalization, dense layer with activation - leaky relu, dropout, given in that order.
I have a few of these layers stacked up (that's a hyperparameter to be tuned, but adding layers adds big complexity and the training is very slow, so I used 3 preactivation layers).
The main part of the model is the preactivation layers. The other layer are used only to deal with the dataset.
1. BaseEmbeddingConcatenator concatenates the base dataset, as I keep it split in user_id, movie_id, movie_features and user_context
2. MergeDatasets concatenated the trailer features if present to the dataset.
To run model training just run 
```shell
python src/main.py --set-your-arguments...
```
You can view the arguments and their description in main.py.
For example if you want to train model with large batch size you can run
```shell
python src/main.py --batch_size=1024
```

## Evaluation Metrics:
To get metrics results please run and set parameters according to which model you're testing. Note:you need to have prediction calculated to run this. If not, run the following prior to running the metrics calculation:
```shell
python src/evaluation/calculate_predictions.py --filtered_dataset=False --aggregation_method="max" --feature_reduction="pca" --include_trailers=True
```
Then you will get a `predictions.npy` file in your model directory which will be used to calculate metrics.
 ```shell
python src/evaluation/get_metrics_main.py --filtered_dataset=False --aggregation_method="max" --feature_reduction="pca" --include_trailers=True
```

There are 3 sets of metrics: accuracy based metrics, ranking based metrics and general performance metrics.
Note: The base dataset is split in 3 parts: train, dev, test. All metrics are calculated on test set.

#### Accuracy based metrics
For all of these metrics I take as labels the true ratings of that the user has provided for a specific movie.
The predictions are the predictions of the rating for the movie given context for the user and movie.
1. Mean Squared Error (MSE):
- Calculation: The MSE is calculated by taking the average of the squared differences between the true target values and the predicted values. The squared differences penalize larger errors more heavily.
- Interpretation: A lower MSE indicates that the model's predictions are closer to the true values, implying better performance. It measures the average squared deviation between predictions and actual values.

2. Root Mean Squared Error (RMSE):
- Calculation: The RMSE is calculated as the square root of the MSE. It provides a measure of the average magnitude of the errors in the same units as the target variable.
- Interpretation: Similar to MSE, a lower RMSE value signifies better model performance. It is a more interpretable metric because it is in the same units as the target variable, making it easier to understand the magnitude of prediction errors.

3. Mean Absolute Error (MAE):
- Calculation: The MAE is computed by taking the average of the absolute differences between the true target values and the predicted values. Unlike MSE, MAE does not square the errors.
- Interpretation: MAE provides a measure of the average absolute deviation between predictions and actual values. Like MSE and RMSE, a lower MAE indicates better model performance. It is less sensitive to outliers compared to MSE.

#### Ranking based metrics
In order to frame the problem as a ranking problem, for each user I take my top k predicted ratings and their corresponding labels, so all the metrics are calculated on this subset of the dataset, moreover, I am giving them a slight advantage because I am taking my top best results. 
Note: these metrics are calculated from the test set which has the true labels so unlike regular recommendation system, where these would be calculated based on user sections, I calculate them by collecting the topK predictions and I calculate metrics on them (I consider the ordered list of these topK predictions as my true labels, so results might seem optimistic).
1. Mean Reciprocal Rank (MRR):
- Computation: MRR measures the effectiveness of a ranking system. It is computed by taking the reciprocal of the rank of the first relevant item in a ranked list and then averaging these reciprocals over multiple queries.
- Interpretation: Higher MRR values indicate better-ranking performance, with 1 being the highest possible score. 

2. Mean Average Precision (MAP):
- Computation: MAP calculates the average precision (AP) for each query and then averages these values over all queries. AP measures the precision of the retrieved results at each relevant document and is particularly useful when dealing with variable-sized result sets.
- Interpretation: A higher MAP indicates better precision in the ranked results. MAP ranges from 0 to 1, with 1 being the best possible score. 

3. Normalized Discounted Cumulative Gain (NDCG):
- Computation: NDCG evaluates the quality of ranked lists by considering both the relevance and the position of items in the list. It compares the cumulative gain achieved by the ranked list to the ideal cumulative gain.
- Interpretation: NDCG values range from 0 to 1, with 1 indicating a perfect ranking. Higher NDCG values imply better quality rankings, accounting for both relevance and ranking position. 
- I have a few versions of this, scaling the value of the good ratings, to get a more realistic value and to make it "harder" for my model to get a good result.

Next 2 metrics are used to measure similarity between lists as compared to the last metrics that were purely ranking based.
4. Kendall's Tau:
- Computation: Kendall's Tau measures the correlation or concordance between two rankings. It counts the number of concordant and discordant pairs in two rankings to compute a correlation coefficient.
- Interpretation: Kendall's Tau ranges from -1 to 1, where 1 indicates perfect agreement, 0 suggests no correlation, and -1 implies perfect disagreement between the rankings. It assesses the similarity between two rankings.

5. Rank-Biased Overlap (RBO):
- Computation: RBO measures the similarity between two ranked lists while considering both the overlap and the rank positions of common items. It introduces a user-defined "patience" parameter to control the sensitivity to ranking depth.
- Interpretation: RBO provides a value between 0 and 1, where higher values indicate greater similarity between the ranked lists. 

#### General performance metrics
For each user I take my top k predicted ratings and their corresponding labels, so all the metrics are calculated on this subset of the dataset.
1. Coverage:
- Computation: Coverage measures the proportion of unique items or entities that have been recommended by a recommendation system out of the entire item space. It assesses the system's ability to recommend a diverse set of items.
- Interpretation: Higher coverage values indicate that the recommendation system suggests a broader range of items, potentially reaching a larger audience. It is useful for ensuring that all items are exposed to users and is often used in conjunction with other metrics to balance diversity and relevance.

2. Personalization:
- Computation: Personalization measures how tailored or unique the recommendations are for each user. It can be calculated using various techniques, such as entropy-based metrics or diversity measures, to evaluate the dissimilarity of recommendations across users. I am using cosine similarity.
- Interpretation: A higher personalization score implies that recommendations are highly customized for each user, offering unique and relevant suggestions. Personalization is crucial in providing a satisfying user experience, as it ensures that users receive recommendations that match their individual preferences and needs.

3. Intra-list Similarity:
- Computation: Intra-list similarity evaluates the diversity or similarity of items within a recommended list or set. It calculates a similarity score between items in the list, often using measures like Jaccard similarity, cosine similarity, or others. I am using cosine similarity.
- Interpretation: Higher intra-list similarity indicates that the items in a recommendation list are more similar to each other. Lower values imply greater diversity among the recommended items. Intra-list similarity is important for balancing diversity and relevance; overly similar recommendations may not be as valuable to users seeking variety
- it is calculated based on the dataset loaded which is defined by arguments set in `src/evaluation/get_metrics_main.py`. So similarity is calculated on all features.

## Model Optimization and Future Work:
The main point to test is the dataset with and without trailers. I've split testing in a few sections.
Then I also test based on aggregation methods (min, max, avg) for the features extracted from each keyframe from the trailer.
Another point of testing is reduction of feature size (pca)
For the base dataset trained both with the whole dataset and the filtered dataset. Filtered dataset includes movie user pairs that actually have trailers, so it's the trailer dataset without trailer features. This is to ensure that the base model doesn't outperform the model that incorporates trailer features only because it's trained on more data.

                                        include_trialers
                                        /               \
                                       /                 \
                                    False                 True
                                  /       \                   \
                                 /         \                   \
                              whole     filtered         aggregation methods
                                                          /       |       \
                                                         /        |        \
                                                       min       max       avg
                                                         \        |        /
                                                          \       |       /    
                                                           \      |      /
                                                      feature reduction methods
                                                          /       |       \
                                                         /        |        \
                                                       PCA       K-PCA     T-SNE                                                               

## Hyperparameter testing: 
 - epochs: [ 25 ]
 - batch_size: [ 16, 32, 64, 128 ]
 - preactivation_layers: [ 3, 7 ]
 - dense_units: [ 64, 128, 256, 512 ]
 - dropout: [ 0.1, 0.15, 0.2 ]
 - embedding_size: [ 30 ]
You can view the results in `model/hyperparameter_tests`. The small models were trained on the filtered dataset, with no trailer features, based on this I picked out the best hyperparameters and trained the rest of the models. 
They were trained on 10K data, which is ~7.5% of the whole dataset. The reason for this is the computational cost, as it takes a long time to train on whole dataset. 
My winning combination was: 
 - epochs: [ 20 ]
 - batch_size: [ 64 ]
 - preactivation_layers: [ 3 ]
 - dense_units: [ 256 ]
 - dropout: [ 0.2 ]
 - embedding_size: [ 30 ]
Future work could involve [add]