import numpy as np
import tensorflow as tf


def unpack_into_TensorArray(input_, axis, size=None):
    """
    Unpack a given tensor along a given axis into a TensorArray
    """

    shape = input_.get_shape().as_list()
    rank = len(shape)
    dtype = input_.dtype
    array_size = shape[axis] if not shape[axis] is None else size

    if array_size is None:
        raise ValueError("Cannot create TensorArray with size None\n")

    array = tf.TensorArray(dtype=dtype, size=array_size)
    dim_permutation = [axis] + range(1, axis) + [0] + range(axis + 1, rank)
    unpack_axis_major_value = tf.transpose(input_, dim_permutation)
    full_array = array.unstack(unpack_axis_major_value)

    return full_array


def stack_into_tensor(array, axis):
    """
    Stack a TensorArray into a tensor along a given axis
    """

    stacked_tensor = array.stack()
    shape = stacked_tensor.get_shape()
    rank = len(shape)

    dim_permutation = [axis] + range(1, axis) + [0] + range(axis + 1, rank)
    correct_shape_tensor = tf.transpose(stacked_tensor, dim_permutation)

    return correct_shape_tensor
