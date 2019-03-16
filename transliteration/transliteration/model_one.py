import tensorflow as tf
from tensorflow import keras as tfk

from tensorflow.contrib.seq2seq import monotonic_attention
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
                 attention='bahdanau',
                 attention_size):
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.attention = attention
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
        output, forward_state, backward_state = self.encoder(embedded)
        return output, backward_state


class CombinedEncoder(Encoder):
    def __init__(self, config: Config, base_encoder: Encoder):
        super(CombinedEncoder, self).__init__(config)
        self.base_encoder = base_encoder

    def call(self, inputs):
        base_out, base_state = self.base_encoder(inputs)
        this_out, _, this_state = self.encoder(self.embedding(inputs))
        return tf.concat([base_out, this_out], axis=-1), tf.concat([base_state, this_state], axis=-1)


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

        if config.attention is None:
            self.attention = None
        elif config.attention == 'bahdanau':
            self.attention = BahdanauAttention(config.attention_size)
        elif config.attention == 'monotonic_bahdanau':
            self.attention = BahdanauMonotonicAttention(config.attention_size)

    def call(self, inputs, states, encoder_output, training=False):
        decoder_hidden, previous_alignments = states
        inputs = self.embedding(inputs)
        if self.attention is not None:
            context, alignments = self.attention([decoder_hidden,
                                                  encoder_output,
                                                  previous_alignments])
            inputs = tf.concat([inputs, context], axis=-1)
        inputs = tf.expand_dims(inputs, axis=1)  # we always only run one timestep
        output, state = self.decoder(inputs, initial_state=decoder_hidden)
        output = self.output_layer(output)
        return output, (state, alignments)

    def make_initial_state(self, encoder_state, encoder_output):
        decoder_hidden = self.initial_state_layer(encoder_state)
        alignments = self.attention.calculate_initial_alignments(encoder_output)
        return decoder_hidden, alignments


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

    def calculate_initial_alignments(self, encoder_out):
        return tf.zeros(encoder_out.shape[:2])  # not actually used

    def call(self, inputs):
        decoder_hidden, encoder_out, previous_alignments = inputs
        decoder_hidden = tf.expand_dims(decoder_hidden, 1)
        weights = self.V(tf.nn.tanh(self.W1(decoder_hidden) + self.W2(encoder_out)))
        weights = tf.nn.softmax(weights, axis=1)
        return tf.reduce_sum(weights * encoder_out, axis=1), tf.squeeze(weights, axis=2)

    def compute_output_shape(self, input_shape):
        # cut out the time dimension of the second input
        return [input_shape[1][0], input_shape[1][2]]


class BahdanauMonotonicAttention(BahdanauAttention):
    def calculate_initial_alignments(self, encoder_out):
        oh = tf.one_hot(0, encoder_out.shape[1])
        return tf.tile(tf.expand_dims(oh, 0), [encoder_out.shape[0], 1])

    def call(self, inputs):
        decoder_hidden, encoder_out, previous_alignments = inputs
        decoder_hidden = tf.expand_dims(decoder_hidden, 1)
        weights = self.V(tf.nn.tanh(self.W1(decoder_hidden) + self.W2(encoder_out)))
        weights = tf.nn.sigmoid(tf.squeeze(weights, axis=2))
        weights = monotonic_attention(weights, previous_alignments, 'recursive')
        return tf.reduce_sum(tf.expand_dims(weights, 2) * encoder_out, axis=1), weights
