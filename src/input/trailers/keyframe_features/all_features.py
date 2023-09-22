import argparse
import glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip

from src.input.trailers.keyframe_features.process_trailer import ProcessTrailer
from src.input.trailers.keyframe_features.youtube_downloader import YouTubeDownloader
from src.input.trailers.util.file_util import FileUtil
from src.input.trailers.util.trailer_util import TrailerUtil
from src.utils.utils import PROJECT_DIR, TRAILER_DATA, YOUTUBE_DATA
from external.keyframe_extraction.viretpipeline.prod.export import extract_keyframes_from_viret

# Note: this is a "hack" used by the viretpipeline, so,
# I have to use it in order for the yaml file to recognise all the dependencies
import external.keyframe_extraction.viretpipeline.prod.video_manipulation
import external.keyframe_extraction.viretpipeline.prod.classification
import external.keyframe_extraction.viretpipeline.prod.scene_detection
import external.keyframe_extraction.viretpipeline.prod.keyframe_selection
import external.keyframe_extraction.viretpipeline.prod.feature_extraction

parser = argparse.ArgumentParser()
# Model parameters
parser.add_argument("--max_duration", default=5, type=int, help="Max duration of trailers you want to process.")


class GetAllFeatures:
    @staticmethod
    def run(max_duration: int):
        # df = pd.read_csv(YOUTUBE_DATA.joinpath("problem.csv"))
        df = pd.read_csv(PROJECT_DIR.joinpath("ml-20m-youtube", "pt_5_ml-youtube.csv"))
        for idx, row in df.iterrows():
            if GetAllFeatures.conditions_to_skip_trailer(row): continue
            start = time.time()
            GetAllFeatures.run_for_trailer(row, max_duration)
            print((time.time() - start) / 60)

    @staticmethod
    def conditions_to_skip_trailer(row):
        return GetAllFeatures.is_trailer_already_processed(row) \
            or GetAllFeatures.trailer_in_ignore(row.movieId)

    @staticmethod
    def is_trailer_already_processed(row):
        movie_name = TrailerUtil.get_movie_name(row.title)
        output_path = TrailerUtil.get_output_path(row.movieId, movie_name)

        return output_path.exists() and output_path.joinpath("trailer_features.npy").exists()

    @staticmethod
    def trailer_in_ignore(movie_id):
        path_ignore = TRAILER_DATA.joinpath(".ignore.npy")
        elements_to_ignore = np.load(f"{path_ignore.__str__()}")
        return movie_id in elements_to_ignore

    @staticmethod
    def run_for_trailer(row, max_duration: int):
        movie_name = TrailerUtil.get_movie_name(row.title)
        output_path = TrailerUtil.get_output_path(row.movieId, movie_name)
        download_successful = YouTubeDownloader.download_trailer(row.youtubeId, row.movieId, movie_name, output_path)
        if download_successful:
            full_trailer_name = glob.glob(os.path.join(output_path, '*.mp4'))[0]
            trailer_name = full_trailer_name.split("/")[-1]
            duration = GetAllFeatures.get_video_duration_in_minutes(output_path.joinpath(full_trailer_name))
            if not duration or duration > max_duration:
                TrailerUtil.update_ignore_file(row.movieId)
                FileUtil.delete_directory_recursive(output_path)
                print(f"Trailer for {movie_name} too long `{duration}` min")
                return
            GetAllFeatures.get_keyframes(output_path, trailer_name)
            GetAllFeatures.get_features(movie_name, output_path, row)

            print(f"Done with {movie_name}")

    @staticmethod
    def get_video_duration_in_minutes(path: Path):
        try:
            clip = VideoFileClip(path.__str__())
            duration = clip.duration
            clip.close()
            return duration / 60
        except Exception as e:
            print("Error:", e)
            return None

    @staticmethod
    def get_keyframes(trailer_dir: Path, trailer_name: str):
        extract_keyframes_from_viret(trailer_dir.joinpath(trailer_name),
                                     trailer_dir.joinpath("transnet_output").__str__())

        FileUtil.delete_directory_recursive(trailer_dir.joinpath("transnet_output"))
        FileUtil.delete_file(trailer_dir.joinpath(trailer_name))

    @staticmethod
    def get_features(movie_name, output_path, row):
        matrix_features = ProcessTrailer().download_helper(row.movieId, movie_name)
        TrailerUtil.save_tensor(matrix_features, output_path.joinpath("trailer_features.npy"))
        FileUtil.delete_directory_recursive(output_path.joinpath("keyframes"))


if __name__ == "__main__":
    args = vars(parser.parse_args([] if "__file__" not in globals() else None))
    GetAllFeatures().run(args["max_duration"])


# TODO: ignore files only in special error not for example when I lose internet connctivity
