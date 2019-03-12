import tensorflow as tf
import numpy as np

import contextlib

from .script import SCRIPTS


def train_one_batch(*,
                    batch,
                    from_script,
                    to_script,
                    encoder,
                    decoder,
                    optimizer,
                    loss_function):
    with tf.GradientTape() as tape, train_time():
        batch_loss = evaluate_one_batch(batch=batch,
                                        from_script=from_script,
                                        to_script=to_script,
                                        encoder=encoder,
                                        decoder=decoder,
                                        loss_function=loss_function)
        variables = encoder.variables + decoder.variables
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def evaluate_one_batch(*,
                       batch,
                       from_script,
                       to_script,
                       encoder,
                       decoder,
                       loss_function):
    start_token = SCRIPTS[to_script].intern_char('<start>')
    input_seq = batch[from_script]
    output_seq = batch[to_script]
    batch_size = int(input_seq.shape[0])
    max_len = output_seq.shape[1]
    batch_loss = 0
    encoder_output, encoder_state = encoder(input_seq)

    decoder_input = tf.constant(start_token, shape=[batch_size])
    decoder_state = decoder.make_initial_state(encoder_state)
    for t in range(max_len):
        decoder_out, decoder_state = decoder(decoder_input,
                                             decoder_state,
                                             encoder_output)
        decoder_input = output_seq[:, t]
        batch_loss += loss_function(decoder_input, decoder_out)
    return batch_loss


def run_one_epoch(dataset, train, **kwargs):
    loss = 0
    count = 0
    for batch in dataset:
        count += 1
        if train:
            loss += train_one_batch(batch=batch, **kwargs)
        else:
            loss += evaluate_one_batch(batch=batch, **kwargs)
    return loss / count


def greedy_decode(*,
                  encoder_output,
                  encoder_state,
                  decoder,
                  from_script,
                  to_script,
                  max_len=20):
    start_token = SCRIPTS[to_script].intern_char('<start>')
    end_token = SCRIPTS[to_script].intern_char('<end>')
    batch_size = int(encoder_output.shape[0])
    results = []
    done = np.zeros(batch_size, dtype=np.bool)

    decoder_input = tf.constant(start_token, shape=[batch_size])
    decoder_state = decoder.make_initial_state(encoder_state)
    while len(results) < max_len and not np.all(done):
        decoder_out, decoder_state = decoder(decoder_input,
                                             decoder_state,
                                             encoder_output)
        decoder_input = np.argmax(decoder_out, axis=-1)
        decoder_input = np.where(done,
                                 np.zeros_like(decoder_input),
                                 decoder_input)
        done = np.logical_or(done, decoder_input == end_token)
        results.append(decoder_input)
    return np.concatenate([np.expand_dims(r, 1) for r in results],
                          axis=1)


def deintern_decode_results(interned_results, to_script):
    results = []
    for i in range(interned_results.shape[0]):
        result = ''
        for interned_token in interned_results[i]:
            token = SCRIPTS[to_script].deintern_char(interned_token)
            if token == '<end>':
                break
            result += token
        results.append(result)
    return results


def transliterate(*,
                  input_strs,
                  from_script,
                  to_script,
                  encoder,
                  decoder,
                  decoding_method=greedy_decode):
    intern_input_fun = SCRIPTS[from_script].intern_char
    input_seqs = np.asarray([[intern_input_fun(c) for c in input_str]
                             for input_str in input_strs])
    encoder_output, encoder_state = encoder(input_seqs)
    results = decoding_method(encoder_output=encoder_output,
                              encoder_state=encoder_state,
                              decoder=decoder,
                              from_script=from_script,
                              to_script=to_script)
    return deintern_decode_results(results, to_script)


@contextlib.contextmanager
def train_time():
    try:
        yield tf.keras.backend.set_learning_phase(1)
    finally:
        tf.keras.backend.set_learning_phase(0)
