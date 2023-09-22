import os
import shutil
from pathlib import Path


class FileUtil:
    @staticmethod
    def delete_file(path: Path):
        os.unlink(path)

    @staticmethod
    def delete_dir(path: Path):
        os.rmdir(path)

    @staticmethod
    def delete_directory_recursive(path: Path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

        FileUtil.delete_dir(path)
