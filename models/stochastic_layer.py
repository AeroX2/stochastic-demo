import tensorflow as tf
from keras import backend as K
from keras.layers import Layer

class StochasticLayer(Layer):

    def __init__(self, output_dim, bit_size = 8, **kwargs):
        self.output_dim = output_dim
        self.bit_size = bit_size

        print("Stochastic layer using a bit size of {}".format(self.bit_size))
        super(StochasticLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        print(input_shape)

        self.w = self.add_weight(name='kernel',
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='random_normal',
                                 trainable=True);
        self.b = self.add_weight(name='bias',
                                 shape=(self.output_dim,),
                                 initializer='zeros',
                                 trainable=True);

        super(StochasticLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #TODO Normalize the values of x,w,b

        # Create copies of inputs (x, w, b)
        reshaped_x = tf.expand_dims(x, 2)
        x_tiled = tf.tile(reshaped_x, [1, self.input_shape[1], self.bit_size])

        reshaped_w = tf.expand_dims(self.w, 2)
        w_tiled = tf.tile(reshaped_w, [1, 1, self.bit_size])

        #reshaped_b = tf.expand_dims(self.b, 1)
        #b_tiled = tf.tile(reshaped_b, [1, self.bit_size])

        # Converted tiled objects to stochastic bitstreams
        network_shape = tf.shape(x_tiled)
        x_random_floats = tf.random_uniform(network_shape)
        x_bitstream = tf.cast(x_random_floats <= x_tiled, tf.bool)

        w_random_floats = tf.random_uniform(network_shape)
        w_bitstream = tf.cast(w_random_floats <= w_tiled, tf.bool)

        # Stochastic multiply with AND
        multiplication = tf.math.logical_and(x_bitstream, w_bitstream)

        final_row = tf.slice(multiplication, [0, 0, 0], [-1, 1, -1])
        final_shape = tf.shape(final_row)
        for row in range(1, self.output_dim-1):
            row_2 = tf.slice(multiplication, [0, row, 0], [-1, 1, -1])
            select_line = tf.cast(tf.random_uniform(final_shape) <= 0.5, tf.bool)
            final_row = tf.where(select_line, final_row, row_2)

        # TODO Add the bias
        #network_shape = tf.shape(b_tiled)
        #b_random_floats = tf.random_uniform(network_shape)
        #b_bitstream = tf.cast(b_random_floats <= b_tiled, tf.bool)

        final_row = tf.cast(final_row, tf.float32)
        print(final_row)
        reduction = tf.reduce_mean(final_row, 2, keepdims=True)
        print(reduction)

        return reduction

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
