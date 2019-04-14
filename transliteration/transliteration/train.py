import tensorflow as tf
import numpy as np

import contextlib
import tempfile

from .script import SCRIPTS


def train_one_batch(*,
                    batch,
                    from_script,
                    to_script,
                    encoder,
                    decoder,
                    train_encoder=True,
                    train_decoder=True,
                    optimizer,
                    loss_function):
    with tf.GradientTape() as tape, train_time():
        batch_loss = evaluate_one_batch(batch=batch,
                                        from_script=from_script,
                                        to_script=to_script,
                                        encoder=encoder,
                                        decoder=decoder,
                                        loss_function=loss_function)
        variables = []
        if train_encoder:
            variables.extend(encoder.variables)
        if train_decoder:
            variables.extend(decoder.variables)
        variables = [v for v in variables if v.trainable]
        gradients = tape.gradient(batch_loss, variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
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
    encoder_output, encoder_state = encoder(input_seq)

    decoder_input = tf.constant(start_token, shape=[batch_size])
    decoder_state = decoder.make_initial_state(encoder_state, encoder_output)
    step_losses = []
    for t in range(max_len):
        decoder_out, decoder_state = decoder(decoder_input,
                                             decoder_state,
                                             encoder_output)
        decoder_input = output_seq[:, t]
        step_losses.append(loss_function(decoder_input, decoder_out))
    step_losses = tf.stack(step_losses, axis=1)
    lengths = tf.cast(batch['length_{}'.format(to_script)], tf.float32)
    batch_loss = tf.reduce_sum(step_losses, axis=1) / lengths
    batch_loss = tf.reduce_mean(batch_loss)
    return batch_loss


def exercise_encoder_decoder(*,
                             encoder,
                             decoder,
                             to_script):
    encoder_output, encoder_state = encoder(np.zeros([1, 1], np.int64))
    start_token = SCRIPTS[to_script].intern_char('<start>')
    decoder_input = tf.constant(start_token, shape=[1])
    decoder_state = decoder.make_initial_state(encoder_state, encoder_output)
    decoder_out, decoder_state = decoder(decoder_input,
                                         decoder_state,
                                         encoder_output)


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


@contextlib.contextmanager
def train_time():
    try:
        yield tf.keras.backend.set_learning_phase(1)
    finally:
        tf.keras.backend.set_learning_phase(0)


def normal_training_regimen(*,
                            train_data,
                            valid_data,
                            models,
                            from_script,
                            to_script,
                            patience=3):
    with tempfile.TemporaryDirectory() as tempdir:
        best_val_loss = None
        early_stop_checkpoint = None
        checkpoint_obj = None

        e = 0
        best_e = 0
        while e - best_e <= patience:
            e += 1
            loss = run_one_epoch(train_data,
                                 True,
                                 from_script=from_script,
                                 to_script=to_script,
                                 encoder=models['encoder'],
                                 decoder=models['decoder'],
                                 optimizer=models['optimizer'],
                                 loss_function=models['loss_function'])
            val_loss = run_one_epoch(valid_data,
                                     False,
                                     from_script=from_script,
                                     to_script=to_script,
                                     encoder=models['encoder'],
                                     decoder=models['decoder'],
                                     loss_function=models['loss_function'])
            if checkpoint_obj is None:
                # have to do so after the graph is built up by using the models
                checkpoint_obj = models['make_checkpoint_obj']()
            if best_val_loss is None or val_loss < best_val_loss:
                best_e = e
                best_val_loss = val_loss
                early_stop_checkpoint = checkpoint_obj.save(file_prefix='{}/{}'
                                                            .format(tempdir, e))
            print("Epoch {}: Train Loss {:.3f}, Valid Loss {:.3f}"
                  .format(e, loss, val_loss))

        checkpoint_obj.restore(early_stop_checkpoint).assert_consumed()
        val_loss = run_one_epoch(valid_data,
                                 False,
                                 from_script=from_script,
                                 to_script=to_script,
                                 encoder=models['encoder'],
                                 decoder=models['decoder'],
                                 loss_function=models['loss_function'])
        print('Validation loss: {:.3f}'.format(val_loss))
        return val_loss
