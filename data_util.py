#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import glob
import random
import math
import collections
import urllib

import numpy as np

from general_util import *
Examples = collections.namedtuple("Examples", "paths, images, labels, unique_labels, count, steps_per_epoch")

def load_examples(config):
    if not os.path.exists(config.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = get_files_with_ext(config.input_dir, "jpg")# glob.glob(os.path.join(config.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = get_files_with_ext(config.input_dir, "png")# glob.glob(os.path.join(config.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    # Assume the subdirectories containing the input images is their categories/labels.
    labels = map(lambda p: get_subdir(p,config.input_dir),input_paths)
    labels, unique_labels = labels_to_one_hot_vector(labels)

    with tf.name_scope("load_examples"):
        # path_queue = tf.train.string_input_producer(input_paths, shuffle=config.mode == "train")
        # The slice input producer can produce as many queus as the user wants.
        labels = tf.constant(np.array(labels,dtype=np.bool))
        input_queue = tf.train.slice_input_producer([input_paths, labels],shuffle=config.mode == "train")
        path_queue = input_queue[0]
        label_queue = tf.to_float(input_queue[1])

        # reader = tf.WholeFileReader()
        # paths, contents = reader.read(path_queue)
        # Can't use whole file reader, so use tf.read_file instead.
        contents = tf.read_file(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if config.gray_input:
            images = tf.image.rgb_to_grayscale(raw_input)
        else:
            images = raw_input

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if config.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [config.scale_size, config.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, config.scale_size - config.crop_size + 1, seed=seed)), dtype=tf.int32)
        if config.scale_size > config.crop_size:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], config.crop_size, config.crop_size)
        elif config.scale_size < config.crop_size:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        images = transform(images)

    paths, images, labels = tf.train.batch([path_queue, images, label_queue], batch_size=config.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / config.batch_size))

    return Examples(
        paths=paths,
        images=images,
        labels=labels,
        unique_labels=unique_labels,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def deprocess(image, config):
    if config.aspect_ratio != 1.0:
        # upscale to correct aspect ratio
        size = [config.crop_size, int(round(config.crop_size * config.aspect_ratio))]
        image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)
    return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

def labels_to_one_hot_vector(labels):
    unique_labels = list(set(labels))

    if all(label.isdigit() for label in unique_labels):
        unique_labels = sorted(unique_labels, key=lambda label: int(label))
    else:
        unique_labels = sorted(unique_labels)
    num_unique_labels = len(unique_labels)
    label_vector_dict = {}
    for i, label in enumerate(unique_labels):
        one_hot_vector = np.zeros(num_unique_labels)
        one_hot_vector[i] = 1
        label_vector_dict[label] = one_hot_vector
    ret = [label_vector_dict[label] for label in labels]
    return ret, unique_labels


def one_hot_vector_to_labels(ohv, unique_labels):
    if ohv.ndim == 1:

        label_indices = np.argmax(ohv)
        labels = unique_labels[label_indices]
        return labels
    elif ohv.ndim == 2:
        label_indices = np.argmax(ohv, axis=1)
        labels = [unique_labels[i] for i in label_indices]
        return labels


def save_results(fetches, image_dir, unique_labels, step=None):
    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path))
        fileset = {"name": name, "step": step}
        for kind in ["inputs"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "w") as f:
                f.write(contents)
        for kind in ["labels", "outputs"]:
            contents = fetches[kind][i]
            labels = one_hot_vector_to_labels(contents, unique_labels)
            fileset[kind] = labels
        filesets.append(fileset)
    return filesets

def append_index(filesets, config, step=False):
    index_path = os.path.join(config.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><meta content=\"text/html;charset=utf-8\" http-equiv=\"Content-Type\"><meta content=\"utf-8\" http-equiv=\"encoding\"><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")
        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs"]:
            index.write("<td><img src=\"images/%s\"></td>" % urllib.quote(fileset[kind]))
        for kind in ["labels", "outputs"]:
            index.write("<td>%s</td>" % fileset[kind])

        index.write("</tr>")
    return index_path