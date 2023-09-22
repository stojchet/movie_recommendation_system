import traceback

import numpy as np
from pytube import YouTube

from src.input.trailers.util.trailer_util import TrailerUtil
from src.utils.utils import TRAILER_DATA

BASE_URL = "https://www.youtube.com/watch?v="


class YouTubeDownloader:
    @staticmethod
    def __get_url(specific_code):
        return BASE_URL + specific_code

    @staticmethod
    def download_trailer(youtubeId, movie_id, movie_name, output_path):
        try:
            link = YouTubeDownloader.__get_url(youtubeId)
            youtube = YouTube(link, use_oauth=True, allow_oauth_cache=True).streams.get_highest_resolution()
            youtube.download(output_path=output_path)
            return True
        except Exception:
            print(f"An error has occurred with movie {movie_id}:\t{movie_name}")

            traceback.print_exc()
            with open(TRAILER_DATA.joinpath(".no_trailer.txt"), "a+") as file:
                content = f"{movie_id}_{movie_name}"
                if content in file.read().split("\n"):
                    return False
                file.write(f"{content}\n")

            with open(TRAILER_DATA.joinpath(".errors.txt"), "a") as file:
                file.write(f"{movie_id}_{movie_name}\n")
                file.write(traceback.format_exc() + "\n\n")

            TrailerUtil.update_ignore_file(movie_id)

            return False
