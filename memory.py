import tensorflow as tf
import numpy as np
import math


class Memory(object):

    def __init__(self, word_num=128, word_size=20, read_heads=1, batch_size=1):
        """
        Contructure a memory matrix with read heads and a write head

        Parameters:
        -------------
            word_num:       the number of memory locations
            word_size:      the size of each location
            read_heads:     the number of read heads that can read simultaneously from the memory
            write_heads:    default 1
            batch_size:     the size of the input data batch
        """
        self.word_num = word_num
        self.word_size = word_size
        self.read_heads = read_heads
        self.write_heads = 1
        self.batch_size = batch_size

    def init_memory(self):
        """
        Returns the initial value for the memory matrix
        This 4 tensors represents the states of the memory
        """
        return (
            tf.truncated_normal([self.batch_size, self.word_num, self.word_size], mean=0.5, stddev=0.2),    # The Memory Matrix
            tf.fill([self.batch_size, self.word_num, 1], 1e-6),                                             # initialization for "Write" weighting
            tf.truncated_normal(self.batch_size, self.word_num, self.read_heads], mean=0.4, stddev=0.1),    # initialization for "Read" weighting
            tf.fill([self.batch_size, self.word_size, self.read_heads], 1e-6)                               # read vectors
        )

    def write(self, memory_matrix, write_weighting, key, strength, interpolation_gate, shift_weighting, gamma, add_vector, erase_vector):
        """
        Implement memory write given the write variables from the previous step

        Parameters:
        ------------
        memory_matrix:      A tensor (batch_size, word_num, word_size)
        write_weighting     A tensor (batch_size, word_num), the write weighting from the last time step
        interpolation_gate:
        shift_weighting:    a Tensor with the shape (batch_size, shift_range*2 + 1, 1)
                            the distribution over the allowed integer shift
        gamma:              a Tensor with the shape (batch_size, 1)
                            scalar to sharpen the final weights
        add_vector:         Tensor (batch_size, word_size)
                            specifications of what to add to memory
        erase_vector:       Tensor (batch_size, word_size)
                            specifications of what to erasae from memory

        """
        content_address_w = self.get_content_addressing(memory_matrix, key, strength)
        gated_weighting = self.apply_interpolation(content_address_w, write_weighting, interpolation_gate)
        conv_shift = self.apply_conv_shift(gated_weighting, shift_weighting)
        new_write_weighting = self.sharp_weights(conv_shift, gamma)

        # Update the Memory
        add_vector = tf.expand_dims(add_vector, 1)
        erase_vector = tf.expand_dims(erase_vector, 1)

        erasing = memory_matrix * (1 - tf.matmul(new_write_weighting, erase_vector))
        adding = tf.matmul(new_write_weighting, add_vector)

        updated_memory = erasing + adding

        return new_write_weighting, updated_memory


    def read(self, memory_matrix, read_weightings, keys, strengths, interpolation_gates, shift_weightings, gammas):
        """
        Implement memory read give the read variables
        """

        content_address_w = self.get_content_addressing(memory_matrix, keys, strengths)
        gated_weightings = self.apply_interpolation(content_address_w, read_weightings, interpolation_gates)
        conv_shift = self.apply_conv_shift(gated_weightings, shift_weightings)
        new_read_weightings = self.sharp_weights(conv_shift, gammas)

        new_read_vectors = tf.matmul(memory_matrix, new_read_weightings, transpose_a=True)

        return new_read_weightings, new_read_vectors


    def get_content_addressing(self, memory, keys, strengths):
        """
        Retrieve a content-based addressing weighting given the keys

        Parameters:
        ------------
        memory:     Tensor = (batch_size, word_num, word_size)
        keys:       Tensor = (batch_size, word_size, # of read_heads)
        strengths:  Tensor = (batch_size, # of read_heads)
        """

        normalized_memory = tf.nn.l2_normalize(memory_matrix, 2)
        normalized_keys = tf.nn.l2_normalize(keys, 1)

        similiarity = tf.matmul(normalized_memory, normalized_keys)
        strengths = tf.expand_dims(strengths, 1)

        return tf.nn.softmax(similiarity * strengths, 1)

    def apply_interpolation(self, content_weights, prev_weights, gate):
        """
        retrieve a location-based addressing given the interpolation_gate

        """
        gate = tf.expand_dims(gate, 1)
        gated_weightings = gate * content_weights + (1.0 - gate) * prev_weights

        return gated_weightings

    def apply_conv_shift(self, gated_weightings, shift_weightings):
        """
        Apply rotation over the gated weights

        size = int(gated_weightings.get_shape()[1])
        kernel_size = int(shift_weightings.get_shape[1])
        kernel_shift = int(math.floor(kernel_size / 2.0))

        def loop(idx):
            if idx < 0:
                return size + idx
            elif idx >= size:
                return idx - size
            else:
                return idx

        kernel = []
        for i in xrange(size):
            indices = [loop(i+j) for j in xrange(kernel_shift, -kernel_shift-1, -1)]
            v_ = tf.transpose(tf.gather(tf.transpose(gated_weightings, perm=(1, 2, 0)), indices), perm=(2, 0, 1))
            kernel.append(tf.reduce_sum(v_ * shift_weightings, 1))

        return tf.stack(kernel, axis=1)
        """
        # TODO: This circular convolution is more efficient than the commented code, but it was harcoded for a batch_size=1
        #       and for a shift_vector with 3 elements.

        gated_weightings = tf.concat(1, [tf.expand_dims(gated_weightings[:,-1,:], axis=-1), gated_weightings, tf.expand_dims(gated_weightings[:,0,:], axis=-1)])

        gated_weightings = tf.expand_dims(gated_weightings, 0)
        shift_weighting = tf.expand_dims(shift_weighting, -1)

        conv = tf.nn.conv2d(
            gated_weightings,
            shift_weighting,
            strides=[1, 1, 1, 1],
            padding="VALID")

        return tf.squeeze(conv, axis=0)

    def sharp_weights(self, conv_shift, gamma):
        """
        Sharpen the final weights
        """
        gamma = tf.expand_dims(gamma, 1)        # shape = (batch_size, 1, # of read_heads)
        powed_conv_w = tf.pow(conv_shift, gamma)
        return powed_conv_w / tf.expand_dims(tf.reduce_sum(powed_conv_w), 1)
