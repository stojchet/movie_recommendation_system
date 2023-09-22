import os

import pandas as pd

from utils.utils import PATH_TO_TRAILER_FEATURES, YOUTUBE_DATA
from src.input.trailers.util.file_util import FileUtil

def find_mp4_files():
    movie_ids = []
    for root, dirs, files in os.walk(PATH_TO_TRAILER_FEATURES):
        for dir in dirs:
            for root1, dirs1, movie_files in os.walk(PATH_TO_TRAILER_FEATURES.joinpath(dir)):
                for file in movie_files:
                    if file.lower().endswith('.mp4'):
                        movie_ids.append(int(dir.split("_")[0]))
                        FileUtil.delete_directory_recursive(PATH_TO_TRAILER_FEATURES.joinpath(dir))

    return movie_ids


def get_leftover_df():
    movie_ids = find_mp4_files()
    df = pd.read_csv(YOUTUBE_DATA.joinpath("ml-youtube.csv"))
    df = df[df["movieId"].isin(movie_ids)]
    df.to_csv(YOUTUBE_DATA.joinpath("leftover-ml-youtube.csv"))


def check():
    count = []
    for root, dirs, files in os.walk(PATH_TO_TRAILER_FEATURES):
        for dir in dirs:
            for root1, dirs1, movie_files in os.walk(PATH_TO_TRAILER_FEATURES.joinpath(dir)):
                if len(movie_files) == 0:
                    count.append(dir)

    print(len(count))
    print(count)


if __name__ == "__main__":
    check()
