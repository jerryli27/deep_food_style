"""
This file contains general utility functions.
"""

import os
import glob
from operator import mul
import numpy as np
from PIL import Image
from typing import Union, List
import math

def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name

def get_subdir(path, parent_dir, strip_slash = True):
    dir = os.path.dirname(path)
    subdir = dir[len(parent_dir):]
    if strip_slash:
        subdir = subdir.strip("/")
    return subdir

def get_files_with_ext(directory, file_ext, search_in_subdir = True):
    if not file_ext.startswith("."):
        file_ext = "." + file_ext
    if search_in_subdir:
        ret = []
        for path, subdirs, files in os.walk(directory):
            for name in files:
                full_file_path = os.path.join(path, name)
                base, ext = os.path.splitext(full_file_path)
                if ext == file_ext:
                    ret.append(full_file_path)
        return ret
    else:
        return glob.glob(os.path.join(directory, "*" + file_ext))

# ----- Copied from old files.
def get_np_array_num_elements(arr):
    # type: (np.ndarray) -> int
    return reduce(mul, arr.shape, 1)

def imread(path, shape=None, bw=False, rgba=False, dtype=np.float32):
    # type: (str, tuple, bool, bool) -> np.ndarray
    """

    :param path: path to the image
    :param shape: (Height, width)
    :param bw: Whether the image is black and white.
    :param rgba: Whether the image is in rgba format.
    :return: np array with shape (height, width, num_color(1, 3, or 4))
    """
    assert not (bw and rgba)
    if bw:
        convert_format = 'L'
    elif rgba:
        convert_format = 'RGBA'
    else:
        convert_format = 'RGB'

    if shape is None:
        return np.asarray(Image.open(path).convert(convert_format), dtype)
    else:
        return np.asarray(Image.open(path).convert(convert_format).resize((shape[1], shape[0])), dtype)

def read_and_resize_images(dirs, height=None, width=None, bw=False, rgba=False):
    # type: (Union[str,List[str]], Union[int,None], Union[int,None], bool, bool) -> Union[np.ndarray,List[np.ndarray]]
    """

    :param dirs: a single string or a list of strings of paths to images.
    :param height: height of outputted images. If height and width are both None, then the image is not resized.
    :param width: width of outputted images. If height and width are both None, then the image is not resized.
    :param bw: Whether the image is black and white
    :param rgba: Whether the image is in rgba format.
    :return: images resized to the specific height or width supplied. It is either a numpy array or a list of numpy
    arrays
    """
    if isinstance(dirs, list):
        images = [read_and_resize_images(d, height, width) for d in dirs]
        return images
    elif isinstance(dirs, str):
        image_1 = imread(dirs)
        # If there is no width and height, we automatically take the first image's width and height and apply to all the
        # other ones.
        if width is not None:
            if height is not None:
                target_shape = (height, width)
            else:
                target_shape = (int(math.floor(float(image_1.shape[0]) /
                                               image_1.shape[1] * width)), width)
        else:
            if height is not None:
                target_shape = (height, int(math.floor(float(image_1.shape[1]) /
                                                       image_1.shape[0] * height)))
            else:
                target_shape = (image_1.shape[0], image_1.shape[1])
        return imread(dirs, shape=target_shape, bw=bw, rgba=rgba)