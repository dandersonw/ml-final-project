import tensorflow as tf
import numpy as np
from tensorflow import keras as tfk

from tensorflow.contrib.seq2seq import monotonic_attention
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional

from . import decode


class Config():
    def __init__(self,
                 *,
                 lstm_size,
                 embedding_size,
                 vocab_size,
                 dropout=0.0,
                 attention='bahdanau',
                 attention_size=None):
        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.attention = attention
        self.attention_size = attention_size
        self.dropout = dropout


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
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, inputs, training=False):
        embedded = self.embedding(inputs)
        output, forward_state, backward_state = self.encoder(embedded)
        output = self.dropout(output)
        backward_state = self.dropout(backward_state)
        return output, backward_state


class CombinedEncoder(Encoder):
    """Concats the output of a base encoder on top of another learned one"""
    def __init__(self, config: Config, base_encoder: Encoder):
        super(CombinedEncoder, self).__init__(config)
        self.base_encoder = base_encoder

    def call(self, inputs):
        this_out, this_state = super(CombinedEncoder, self).call(inputs)
        base_out, base_state = self.base_encoder(inputs)
        return (tf.concat([base_out, this_out], axis=-1),
                tf.concat([base_state, this_state], axis=-1))


class Decoder(tfk.Model):
    def __init__(self, config: Config):
        super(Decoder, self).__init__()
        assert config.attention_size is not None
        self.config = config
        self.embedding = Embedding(config.vocab_size,
                                   config.embedding_size,
                                   mask_zero=True)
        self.initial_state_layer = Dense(config.lstm_size)
        self.decoder = GRU(config.lstm_size,
                           return_state=True)
        self.output_layer = Dense(config.vocab_size)
        self.attention = build_attention(config.attention,
                                         config.attention_size)

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


class StackedEncoderDecoderEncoder(tfk.layers.Layer):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 decoder_script):
        super(StackedEncoderDecoderEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_script = decoder_script

    def call(self, inputs):
        encoder_out, encoder_state = self.encoder(inputs)
        decoder_out, _ \
            = decode.beam_search_decode(encoder_output=encoder_out,
                                        encoder_state=encoder_state,
                                        decoder=self.decoder,
                                        to_script=self.decoder_script,
                                        k_best=1)
        decoder_out = tf.squeeze(decoder_out, axis=-2)
        decoder_out = self.decoder.embedding(decoder_out)
        return (encoder_out, decoder_out), encoder_state


def build_attention(name, size):
    if name.startswith('multiple:'):
        name = name[len('multiple:'):]
        components = [attention_class_for_string(c) for c in name.split(',')]
        return MultipleAttention(size, components)
    else:
        return attention_class_for_string(name)(size)


def attention_class_for_string(name):
    if name is None:
        return None
    elif name == 'bahdanau':
        return BahdanauAttention
    elif name == 'monotonic_bahdanau':
        return BahdanauMonotonicAttention


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


class MultipleAttention(tfk.layers.Layer):
    def __init__(self, attention_size, components):
        assert len(components) > 1
        self.components = [c(attention_size) for c in components]
        super(MultipleAttention, self).__init__()

    def build(self, input_shape):
        key_shape, value_shapes, alignment_shapes = input_shape
        for component, value_shape, alignment_shape in zip(self.components, value_shapes, alignment_shapes):
            component.build((key_shape, value_shape, alignment_shape))
        super(MultipleAttention, self).build(input_shape)

    def calculate_initial_alignments(self, attention_values):
        assert len(attention_values) == len(self.components)
        return [c.calculate_initial_alignments(value)
                for c, value in zip(self.components, attention_values)]

    def call(self, inputs):
        attention_key, attention_values, previous_alignments = inputs
        assert len(attention_values) == len(self.components)
        assert len(previous_alignments) == len(self.components)

        outputs = []
        alignments = []
        for component, value, alignment in zip(self.components,
                                               attention_values,
                                               previous_alignments):
            output, alignment = component((attention_key, value, alignment))
            outputs.append(output)
            alignments.append(alignment)

        return tf.concat(outputs, axis=-1), alignments


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred)
    return tf.reduce_mean(loss_ * mask)
    # return loss_ * mask
