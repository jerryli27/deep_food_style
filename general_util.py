"""
This file contains general utility functions.
"""

import os
import glob

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
