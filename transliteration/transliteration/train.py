import tensorflow as tf
import numpy as np

from . import data


def train_one_batch(*,
                    batch,
                    input_key,
                    output_key,
                    encoder,
                    decoder,
                    optimizer,
                    loss_function,
                    start_token):
    with tf.GradientTape() as tape:
        batch_loss = evaluate_one_batch(batch=batch,
                                        input_key=input_key,
                                        output_key=output_key,
                                        encoder=encoder,
                                        decoder=decoder,
                                        loss_function=loss_function,
                                        start_token=start_token)
        variables = encoder.variables + decoder.variables
        gradients = tape.gradient(batch_loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss


def evaluate_one_batch(*,
                       batch,
                       input_key,
                       output_key,
                       encoder,
                       decoder,
                       loss_function,
                       start_token):
    input_seq = batch[input_key]
    output_seq = batch[output_key]
    batch_size = int(input_seq.shape[0])
    max_len = output_seq.shape[1]
    batch_loss = 0
    encoder_output, encoder_state = encoder(input_seq)
    decoder_input = tf.constant(start_token, shape=[batch_size])
    decoder_state = encoder_state
    for t in range(max_len):
        decoder_out, decoder_state = decoder(decoder_input,
                                             decoder_state,
                                             encoder_output)
        decoder_input = output_seq[:, t]
        batch_loss += loss_function(decoder_input, decoder_out)
    batch_loss /= batch_size
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
                  start_token,
                  end_token,
                  max_len=20):
    batch_size = int(encoder_output.shape[0])
    results = []
    done = np.zeros(batch_size, dtype=np.bool)
    decoder_input = tf.constant(start_token, shape=[batch_size])
    decoder_state = encoder_state
    while len(results) < max_len and not np.all(done):
        decoder_out, decoder_state = decoder(decoder_input,
                                             decoder_state,
                                             encoder_output)
        decoder_input = np.argmax(decoder_out, axis=-1)
        done = np.logical_or(done, decoder_input == end_token)
        decoder_input = np.where(done,
                                 np.zeros_like(decoder_input),
                                 decoder_input)
        results.append(decoder_input)
    return np.concatenate([np.expand_dims(r, 1) for r in results],
                          axis=1)


def deintern_decode_results(interned_results, deintern_fun):
    results = []
    for i in range(interned_results.shape[0]):
        result = ''
        for interned_token in interned_results[i]:
            token = deintern_fun(interned_token)
            if token == '<end>':
                break
            result += token
        results.append(result)
    return results


def transliterate_single(*,
                         input_str,
                         intern_input_fun,
                         deintern_output_fun,
                         encoder,
                         decoder):
    input_seq = np.asarray([[intern_input_fun(c) for c in input_str]])
    encoder_output, encoder_state = encoder(input_seq)
    results = greedy_decode(encoder_output=encoder_output,
                            encoder_state=encoder_state,
                            decoder=decoder,
                            start_token=data.intern_katakana_char('<start>'),
                            end_token=data.intern_en_char('<end>'))
    return deintern_decode_results(results, deintern_output_fun)
