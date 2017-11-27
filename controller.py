import tensorflow as tf
import numpy as np

class BaseController(object):

    def __init__(self, input_size, output_size, word_size, read_heads=1, write_heads=1, shift_range=1, batch_szie=1):
        """
        Parameters:
        ------------
            input_size:      the dimension of the input vector
            output_size:     the dimension of the output vector
            read_heads: the number of read heads in the associative external memory
            word_size:      the size of word in the associative external memory
            shift_range:    allowed integer shifts
            batch_size:     the size of the input data batch
        """

        self.input_size = input_size
        self.output_size = output_size
        self.word_size = word_size
        self.read_heads = read_heads
        self.write_heads = write_heads
        self.shift_range = shift_range
        self.batch_size = batch_size

        # indicates if the internal neural network is recurrent
        # by the existence of recurrent_update and get_state methods
        has_recurrent_update = callable(getattr(self, "recurrent_update", None))
        has_get_state = callable(getattr(self, "get_state", None))
        self.has_recurrent_nn = has_recurrent_update and has_get_state

        # The actual size of the nn input after flattenning and concatenating the input vector with the previously read vectors from memory
        self.nn_input_size = self.read_heads * self.word_size + self.input_size  # a single controller input: [x_t, r1_t-1, r2_t-1, ... r_R_t-1]  R: read_heads

        # interface vector: size of the output vector from the Controller that defines interactions with external memory at current time step
        # Heads: (X reading heads + 1 writing head) each head first produces a length word_size key vector k_t, and generates 3 scalar value for strength gate, gamma
        # Then, each head generates shift_range, and erase and add vector with length word_size.
        # For each head: key + strength + gate + gamma + shift | erase and add vector
        self.interface_vector_size = (self.word_size + 3 + (2 * self.shift_range + 1)) * (self.read_heads + self.write_heads) + 2 * self.word_size

        # Define networks variables
        with tf.name_scope("lstm_controller"):
            # self.network_vars()
            # Define variables needed by the internal network (using LSTM as our neural network)
            self.lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=100)
            self.state = tf.Variable(tf.zeros([self.batch_size, 100]), trainable=False)
            self.output = tf.Variable(tf.zeros([self.batch_size, 100]), trainable=False)
            self.nn_output_size = None

            with tf.variable_scope("shape_inference"):
                input_ = tf.convert_to_tensor(np.zeros([self.batch_size, self.nn_input_size], dtype=tf.float32))
                output_vector = self.lstm_cell(inputs=input_), state=(self.output, self.state))
                shape = output_vector.get_shape().as_list()

                if len(shape) > 2:
                    raise ValueError("Expected the neural network to output a 1D vector, but got %dD" % (len(shape) - 1))
                else:
                    self.nn_output_size = shape[1]

                self.initials()

        def initials(self):
            """
            Set the initial values of the Controller transformation wrghts matrices
            this method can be overwritten to use a different initialization scheme
            """
            # defining internal weights of the controller
            self.interface_weights = tf.Variable(
                initial_value=tf.random_normal([self.nn_output_size, self.interface_vector_size], stddev=0.1),
                name='interface_weights'
            )
            self.nn_output_weights = tf.Variable(
                initial_value=tf.random_normal([self.nn_output_size, self.output_size], stddev=0.1),
                name='nn_output_weights'
            )

        def parse_interface_vector(self, interface_vector):
            """
            Parses the flat interface_vector into its various components with their correct shapes

            Parameters:
            ------------
                interface_vector: a Tensor with shape = (batch_size, interface_vector_size),
                                  the flattened interface vector to be parsed

            Return:
            ------------
                parsed: a dictionary with the components of the parsed interface_vector
            """
            parsed = {}

            # Find the relative end positon on the flattened interface vector
            r_key_end = self.word_size * self.read_heads
            r_strengths_end = r_key_end + self.read_heads
            r_gates_end = r_strengths_end + self.read_heads
            r_gamma_end = r_gates_end + self.read_heads
            r_shift_end = r_gamma_end + (self.shift_range * 2 + 1) * self.read_heads

            w_key_end = r_shift_end + self.word_size
            w_strengths_end = w_key_end + 1         # Scalar
            w_gates_end = w_strengths_end + 1       # Scalar
            w_gamma_end = w_gates_end + 1           # Scalar
            w_shift_end = w_gamma_end + (self.shift_range * 2 + 1)

            erase_end = w_shift_end + self.word_size
            write_end = erase_end + self.word_size

            r_keys_shape = (-1, self.word_size, self.read_heads)
            r_scalars_shape = (-1, self.read_heads)
            r_shift_shape = (-1, self.shift_range * 2 + 1, self.read_heads)

            w_key_shape = (-1, self.word_size, 1)
            w_scalars_shape = (-1, 1)
            w_shift_shape = (-1, self.shift_range * 2 + 1, 1)

            write_shape = erase_shape = (-1, self.word_size)

            # parsing the vector into its individual components
            parsed['read_keys'] = tf.reshape(interface_vector[:, :r_keys_end], r_keys_shape)
            parsed['read_strengths'] = tf.nn.softplus(tf.reshape(interface_vector[:, r_keys_end:r_strengths_end], r_scalars_shape))+1
            parsed['read_gates'] = tf.sigmoid(tf.reshape(interface_vector[:, r_strengths_end:r_gates_end], r_scalars_shape))
            parsed['read_gammas'] = tf.nn.softplus(tf.reshape(interface_vector[:, r_gates_end:r_gamma_end], r_scalars_shape))+1
            parsed['read_shifts'] = tf.nn.softmax(tf.reshape(interface_vector[:, r_gamma_end:r_shift_end], r_shift_shape),dim=1)

            parsed['write_key'] = tf.reshape(interface_vector[:, r_shift_end:w_key_end], w_key_shape)
            parsed['write_strength'] = tf.nn.softplus(tf.reshape(interface_vector[:, w_key_end:w_strengths_end], w_scalars_shape))+1
            parsed['write_gate'] = tf.sigmoid(tf.reshape(interface_vector[:, w_strengths_end:w_gates_end], w_scalars_shape))
            parsed['write_gamma'] = tf.nn.softplus(tf.reshape(interface_vector[:, w_gates_end:w_gamma_end], w_scalars_shape)) + 1
            parsed['write_shift'] = tf.nn.softmax(tf.reshape(interface_vector[:, w_gamma_end:w_shift_end], w_shift_shape),dim=1)

            parsed['erase_vector'] = tf.sigmoid(tf.reshape(interface_vector[:, w_shift_end:erase_end], erase_shape))
            parsed['write_vector'] = tf.reshape(interface_vector[:, erase_end:write_end], write_shape)

            return parsed

        def process_input(self, input_, last_read_vectors, state=None):
            """
            Processes input data through the Controller and
            returns the pre-output and interface_vector

            Parameters:
            -------------
                input_: (batch_size, input_size)
                last_read_vectors: (batch_size, word_size, read_heads)
                state: Tuple
                    state vectors if network is recurrent

            Returns:
            -------------
                pre_output: (batch_size, output_size)
                parsed_interface: dict
            """

            flat_read_vectors = tf.reshape(last_read_vectors, (-1, self.word_size * self.read_heads))
            complete_input = tf.concat(values=[input_, flat_read_vectors], axis=1)
            nn_output, nn_state = None, None

            if self.has_recurrent_nn:
                nn_output, nn_state = self.network_op(complete_input, state)
            else:
                nn_output = self.network_op(complete_input)

            pre_output = tf.matmul(nn_output, self.nn_output_weights)
            interface = tf.matmul(nn_output, self.interface_weights)
            parsed_interface = self.parse_interface_vector(interface)

            if self.has_recurrent_nn:
                return pre_output, parsed_interface, nn_state
            else:
                return pre_output, parsed_interface
