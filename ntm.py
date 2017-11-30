import os
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from memroy import Memory
from controller import Controller
import utils

class NTM(object):
    """
    Implementation of Neural Turning Machine

    Controller:
        - Accept inputs: X_t = [x_t, r_1, r_2, ... (read vector from t-1 memory matrix)]
        - Then, the controller will emit an interface that defines its interactions with the memory at the current time t
        - Using LSTM architecture inside the controller
            i_t = sigmoid( W_i * [X_t, h_t-1, h_l-1] + b_i)
            f_t = sigmoid( W_f * [X_t, h_t-1, h_l-1] + b_f)
        - At each time-step, controller will emit an "output vector" and "interface vector"
            pre_output       =
            interface_vector =
    Memory:
        - contains 4 states, which are [ memort_matrix , write_weighting, read_weightings, read_vectors ]
        - For "write_weighting" : shape = (batch_size, word_num, 1), describe the distribution of write operation
        - For "read_weightings" : shape = (batch_size, word_num, read_heads), describe the distribution of read operation
        - For "read_vectors"    : shape = (batch_size, word_size, read_heads), contains the vectors that being read in the current time step,
                                  these vectors will be the input to the next time step
        -

    """


    def __init__(self, input_size, output_size,
                 memory_word_num=256, memory_word_size=64,
                 read_heads=4, shift_range=1, batch_size=1):
        """
        Parameters:
        ------------
        input_size:
        output_size:
        memory_word_num:    the number on memory locations
        memory_word_size:   the size of each memory word
        read_heads:         the number of the read heads
        shift_range:        allowed integer shifts
        batch_size:
        """

        self.input_size = input_size
        self.output_size = output_size
        self.memory_word_num = memory_word_num
        self.memory_word_size = memory_word_size
        self.read_heads = read_heads
        self.write_head = 1
        self.shift_range = shift_range
        self.batch_size = batch_size

        # initialize the Controller and Memory
        self.controller = Controller(self.input_size, self.output_size, self.memory_word_size, self.read_heads, self.shift_range, self.batch_size)
        self.memory = Memory(self.memory_word_num, self.memory_word_size, self.read_heads, self.batch_size)


        self.input_data = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, input_size], name='input')
        self.targets = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, output_size], name='output')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[1], name='sequence_length')

        self.build_graph()

    def build_graph(self):
        """
        Build the computational graph that performs a step-by-step evaluation of the input data batches
        """

        # Unstack the inputs into TensorArray
        self.unpacked_inputs = utils.unpack_into_TensorArray(self.input_data, axis=1, size=self.sequence_length)  # shape = (1, 10)

        # TensorArray: dynamic-sized, per-time-step, write-once Tensor arrays
        # Using these TensorArray in each loop
        outputs = tf.TensorArray(tf.float32, self.sequence_length)
        read_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        write_weightings = tf.TensorArray(tf.float32, self.sequence_length)
        write_vectors = tf.TensorArray(tf.float32, self.sequence_length)
        key_vectors = tf.TensorArray(tf.float32, self.sequence_length)
        beta_vectors = tf.TensorArray(tf.float32, self.sequence_length)
        shift_vectors = tf.TensorArray(tf.float32, self.sequence_length)
        gamma_vectors = tf.TensorArray(tf.float32, self.sequence_length)
        gates_vectors = tf.TensorArray(tf.float32, self.sequence_length)
        memory_vectors = tf.TensorArray(tf.float32, self.sequence_length)

        controller_state = self.controller.get_state()
        if not isinstance(controller_state, LSTMStateTuple):
            controller_state = LSTMStateTuple(controller_state[0], controller_state[1])

        memory_state = self.memory.init_memory()   # [memory_matrix, write_weighting, read_weightings, read_vectors]
        final_results = None

        with tf.variable_scope("Sequence_Loop") as scope:
            time = tf.constant(0, dtype=tf.int32)

            final_results = tf.while_loop(
                cond=lambda time, *_: time < self.sequence_length,
                body=self._loop_body,
                loop_vars=(
                    time, controller_state, memory_state, outputs,
                    read_weightings, write_weightings, write_vectors, key_vectors, beta_vectors,
                    gamma_vectors, shift_vectors, gates_vectors, memory_vectors
                ),
                parallel_iterations=32,
                swap_memory=True
            )

        dependencies = []
        dependencies.append(self.controller.update_state(final_results[5]))

        with tf.control_dependencies(dependencies):
            self.stacked_output = utils.stack_into_tensor(final_results[2], axis=1)
            # stacked_memory_view and its content is just for debugging purposes
            self.stacked_memory_view = {
                'read_weightings': utils.stack_into_tensor(final_results[3], axis=1),
                'write_weightings': utils.stack_into_tensor(final_results[4], axis=1),
                'write_vectors': utils.stack_into_tensor(final_results[6], axis=1),
                'key_vectors': utils.stack_into_tensor(final_results[7], axis=1),
                'beta_vectors': utils.stack_into_tensor(final_results[8], axis=1),
                'shift_vectors': utils.stack_into_tensor(final_results[9], axis=1),
                'gamma_vectors': utils.stack_into_tensor(final_results[10], axis=1),
                'gates_vectors': utils.stack_into_tensor(final_results[11], axis=1),
                'memory_vectors': utils.stack_into_tensor(final_results[12], axis=1)
            }



    def _loop_body(self, time, controller_state, memory_state, outputs, read_weightings, write_weighting, write_vectors,
                   key_vectors, beta_vectors, gamma_vectors, shift_vectors, gates_vectors, memory_vectors):
        """
        the body of the DNC sequence processing loop

        Parameters:
        ------------
            time:               A tensor
            controller_state:   Tuple
            memory_state:       Tuple
            outputs:            TensorArray
            read_weightings:    TensorArray
            write_weighting:   TensorArray

        Returns:
            A tuple contains all updated arguments
        """

        step_input = self.unpacked_inputs.read(time)

        output_list = self._step_op(step_input, memory_state, controller_state)

        # Update memory state
        updated_memory_state = tuple(output_list[0:4])

        # Update controller state
        updated_controller_state = LSTMStateTuple(output_list[5], output_list[6])

        outputs = outputs.write(time, output_list[4])

        # Collecting memory view for the current step
        read_weightings = read_weightings.write(time, output_list[2])
        write_weighting = write_weighting.write(time, output_list[1])
        write_vectors = write_vectors.write(time, output_list[7])
        key_vectors = key_vectors.write(time, output_list[8])
        beta_vectors = beta_vectors.write(time, output_list[9])
        shift_vectors = shift_vectors.write(time, output_list[10])
        gamma_vectors = gamma_vectors.write(time, output_list[11])
        gates_vectors = gates_vectors.write(time, output_list[12])

        return (
            time+1, updated_memory_state, outputs,
            read_weightings, write_weighting,
            updated_controller_state, write_vectors,
            key_vectors, beta_vectors, shift_vectors, gamma_vectors,
            gates_vectors, memory_vectors
        )


    def _step_op(self, step, memory_state, controller_state=None):
        """
        Perform a step operation on the input step data

        Parameters:
        ------------
            memory_state:       the current memory parameters (tuple)
            controller_state:   the state of the controller if it's recurrent

        Returns:
            output:         Tensor (batch_size, output_size)
            memory_view:    dict
        """

        # Get the last time step's read vectors and be the input of current time step
        last_read_vectors = memory_state[3]

        # After we collect the current step input and last step vectors, we can process these inputs to the Controller
        # "interface" will contain the scalars and vectors that can be used to generate the read/write weightings
        pre_output, interface, nn_state = self.controller.process_step_input(step, last_read_vectors, controller_state)

        write_weighting, memory_matrix = self.memory.write(
            memory_state[0], # 0: memroy matrix
            memory_state[1], # write_weighting
            interface['write_key'],
            interface['write_strength'],
            interface['write_gate'],
            interface['write_shift'],
            interface['write_gamma'],
            interface['add_vector'],
            interface['erase_vector']
        )

        read_weightings, read_vectors = self.memory.read(
            memory_matrix,      # Updated memory matrix
            memory_state[2],
            interface['read_keys'],
            interface['read_strengths'],
            interface['read_gates'],
            interface['read_shifts'],
            interface['read_gammas']
        )

        return [
            # report New Memory State to be updated outside the condition branch
            memory_matrix,
            write_weighting,
            read_weightings,
            read_vectors,

            # v_t
            pre_output,

            # report new state for LSTM
            nn_state[0],
            nn_state[1],

            interface['add_vector'],    #  7
            interface['read_keys'],     #  8
            interface['read_strengths'],#  9
            interface['read_shifts'],   # 10
            interface['read_gammas'],
            interface['read_gates']
        ]
