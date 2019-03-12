import tensorflow as tf
from tensorflow import keras as tfk

from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional


class Config():
    def __init__(self,
                 *,
                 lstm_size,
                 embedding_size,
                 vocab_size,
                 attention_size):
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.attention_size = attention_size


class Encoder(tfk.Model):
    def __init__(self, config: Config):
        super(Encoder, self).__init__()
        self.config = config
        self.embedding = Embedding(config.vocab_size,
                                   config.embedding_size,
                                   mask_zero=True)
        self.encoder = Bidirectional(GRU(config.lstm_size,
                                         return_sequences=True,
                                         return_state=True),
                                     merge_mode='concat')

    def call(self, inputs, training=False):
        embedded = self.embedding(inputs)
        output, foward_state, backward_state = self.encoder(embedded)
        return output, backward_state


class Decoder(tfk.Model):
    def __init__(self, config: Config):
        super(Decoder, self).__init__()
        self.config = config
        self.embedding = Embedding(config.vocab_size,
                                   config.embedding_size,
                                   mask_zero=True)
        self.initial_state_layer = Dense(config.lstm_size)
        self.decoder = GRU(config.lstm_size,
                           return_state=True)
        self.output_layer = Dense(config.vocab_size)
        if config.attention_size is not None:
            self.attention = BahdanauAttention(config.attention_size)
        else:
            self.attention = None

    def call(self, inputs, states, encoder_output, training=False):
        inputs = self.embedding(inputs)
        if self.attention is not None:
            context = self.attention([states, encoder_output])
            inputs = tf.concat([inputs, context], axis=-1)
        inputs = tf.expand_dims(inputs, axis=1)  # we always only run one timestep
        output, state = self.decoder(inputs, initial_state=states)
        output = self.output_layer(output)
        return output, state

    def make_initial_state(self, encoder_state):
        return self.initial_state_layer(encoder_state)


class BahdanauAttention(tfk.layers.Layer):
    def __init__(self, attention_size):
        self.attention_size = attention_size
        super(BahdanauAttention, self).__init__()

    def build(self, input_shape):
        self.W1 = Dense(self.attention_size,
                        use_bias=False,
                        activation=None)
        self.W2 = Dense(self.attention_size,
                        use_bias=False,
                        activation=None)
        self.V = Dense(1, activation=None, use_bias=False)
        self.W1.build(input_shape[0])
        self._trainable_weights = self.W1.trainable_weights
        self.W2.build(input_shape[1])
        self._trainable_weights.extend(self.W2.trainable_weights)
        self.V.build(self.attention_size)
        self._trainable_weights.extend(self.V.trainable_weights)
        super(BahdanauAttention, self).build(input_shape)

    def call(self, inputs):
        decoder_hidden, encoder_out = inputs
        decoder_hidden = tf.expand_dims(decoder_hidden, 1)
        weights = self.V(tf.nn.tanh(self.W1(decoder_hidden) + self.W2(encoder_out)))
        weights = tf.nn.softmax(weights, axis=1)
        return tf.reduce_sum(weights * encoder_out, axis=1)

    def compute_output_shape(self, input_shape):
        # cut out the time dimension of the second input
        return [input_shape[1][0], input_shape[1][2]]
