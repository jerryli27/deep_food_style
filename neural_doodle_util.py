# This file contains functions related to neural doodle. That is we feed in two additional semantic mask layers to
# tell the model which part of the object is what. Using mrf loss and nearest neighbor matching, this technique can
# essentially "draw" according to the mask layers provided.

import tensorflow as tf
from typing import Tuple, Dict

from general_util import *
from neural_util import gramian


def concatenate_mask_layer_tf(mask_layer, vgg_feature_layer):
    # type: (Union[np.ndarray,tf.Tensor], Union[np.ndarray,tf.Tensor]) -> tf.Tensor
    """

    :param mask_layer: mask with shape (num_batch, height, width, num_masks)
    :param vgg_feature_layer: The vgg feature layer with shape (num_batch, height, width, num_features)
    :return: The two layers concatenated in their last dimension.
    """
    return tf.concat(3, [mask_layer, vgg_feature_layer])

def vgg_layer_dot_mask(masks, vgg_layer):
    # type: (Union[np.ndarray,tf.Tensor], Union[np.ndarray,tf.Tensor]) -> tf.Tensor
    """

    :param masks:  mask with shape (num_batch, height, width, num_masks)
    :param vgg_layer: The vgg feature layer with shape (num_batch, height, width, num_features)
    :return: The two layers dotted for each mask and each feature. The returned tensor will have shape
    (num_batch, height, width, num_features * num_masks)
    """
    masks_dim_expanded = tf.expand_dims(masks, 4)
    vgg_layer_dim_expanded = tf.expand_dims(vgg_layer, 3)
    dot = tf.mul(masks_dim_expanded, vgg_layer_dim_expanded)

    batch_size, height, width, num_mask, num_features = map(lambda i: i.value, dot.get_shape())
    dot = tf.reshape(dot, [batch_size, height, width, num_mask * num_features])
    return dot

def masks_average_pool(masks):
    # type: (tf.Tensor) -> Dict[str,tf.Tensor]
    """
    This  function computes the average pool of a given mask to simulate the process of an image being passed through
    the vgg network and the boarders in the image become blurry after convolution layers and pooling layers. (It
    make less sense to apply a perfectly sharp mask to a blurry image.)
    :param masks: The mask to compute average pool over.
    :return: A dictionary with key = layer name and value = the average pooled masked at that layer.
    """
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    ret = {}
    current = masks
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            current = tf.contrib.layers.avg_pool2d(current, kernel_size=[3,3], stride=[1,1],padding='SAME')
        elif kind == 'relu':
            pass
        elif kind == 'pool':
            current = tf.contrib.layers.avg_pool2d(current, kernel_size=[2,2], stride=[2,2],padding='SAME')
        ret[name] = current

    assert len(ret) == len(layers)
    return ret


def gramian_with_mask(layer, masks):
    # type: (Union[np.ndarray,tf.Tensor], tf.Tensor, bool) -> tf.Tensor
    """
    It computes the gramian of the given layer with given masks. Each mask will have its independent gramian for that
    layer.
    :param layer: The vgg feature layer with shape (num_batch, height, width, num_features)
    :param masks: mask with shape (num_batch, height, width, num_masks)
    :return: a tensor with dimension gramians of dimension (num_masks, num_batch, num_features, num_features)
    """
    mask_list = tf.unpack(masks, axis=3) # A list of masks with dimension (1,height, width)

    gram_list = []

    for mask in mask_list:
        mask = tf.expand_dims(mask, dim=3)
        layer_dotted_with_mask = vgg_layer_dot_mask(mask, layer)
        layer_dotted_with_mask_gram = gramian(layer_dotted_with_mask)
        # Normalization is very importantant here. Because otherwise there is no way to compare two gram matrices
        # with different masks applied to them.
        layer_dotted_with_mask_gram_normalized = layer_dotted_with_mask_gram / (tf.reduce_mean(mask) + 0.000001) # Avoid division by zero.
        gram_list.append(layer_dotted_with_mask_gram_normalized)
    grams = tf.pack(gram_list)

    if isinstance(layer, np.ndarray):
        _, _, _, num_features = layer.shape
    else:
        _,_,_,num_features  =  map(lambda i: i.value, layer.get_shape())

    number_colors,_, gram_height, gram_width,  = map(lambda i: i.value, grams.get_shape())

    assert num_features == gram_height
    assert num_features == gram_width
    assert number_colors == len(mask_list)

    return grams