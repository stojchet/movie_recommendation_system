# Movie Recommendation System utilizing Movie Trailers

##### Goal of the project

Create a recommendation system for movies that uses the movie trailer data to produce better recommendations.

##### Project Structure

The project consists of two important parts: the dataset and the model.

## Dataset

### Dataset Preparation

#### Base Dataset
The dataset used in this project is Movielens dataset that has some parsing and modification on top of it. Mainly the modification include: getting the movie genre, adding columns for weekend and time of the day, mean movie rating, extracted the date of release etc. 
After the initial parsing the movie and ratings dataframes are merged on MOVIE_ID and I get one big dataset.

Tensorflow offers an API for working with datasets which provides a smooth and optimized way of working with the datasets. I decided to use this so in ```input/dataset/preprocess/base_dataset/tfTensorDataset.py``` there is an implementation that handles the transformation, and it's also aided by ```pdDataFrameDataset.py``` (Responsible mostly for getting train, dev, test splits) and ```tfTensorDataset.py``` (Converting dataset to a Tensor Dataset).

#### Trailer Enriched Dataset
Fist there is a download script that helps with downloading the trailers from YouTube. There were a lot of trailers that were unavailable so the Dataset size had to also be cut, i.e. remove all movie user pairs that for which the trailer was unavailable.
My goal was to download the video, extract keyframes and extract features for each keyframe. Then save the Tensor matrix of size (#keyframes, feature_space_size). The aggregation is done later, for the purpose of allowing easy testing for aggregation methods.
The keyframe extraction is done via ```Viret Pipeline``` I got from Patrik [paceholder_for_name]
I am using vgg or resnet for extracting features from an image.
Then Just going through all keyframes and extracting features for all in ```input/trailers/keyframe_features/process_trailer.py``` .

### Working with the dataset

The processing part of the dataset should be done as a separate step that will save the prepared for usage datasets. After that there's a wrapper class ```input/dataset/prepared/dataset.py``` that helps with using loading the dataset based on a flag ```"include_trailers"```

## Model

The model is stored in ```model``` package. The ```core``` package contains a full preactivation layer which is a dense layer with all add-ons (normalization, activation and pooling). There are a few input streams that are processed a bit differently. The model consists of a few of those layers then some embeddings for the base dataset and a way of fusing the features together - I chose concatenation.

## Evalutation 

Evaluation is stored in ```evaluation``` package. There are 2 types of metrics being migrated to tf.Metric type



Winning combination: TODO add