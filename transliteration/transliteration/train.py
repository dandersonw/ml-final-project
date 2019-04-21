import tensorflow as tf
import numpy as np

import contextlib
import tempfile

from .script import SCRIPTS
from . import decode, evaluate, model_setup


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
            variables.extend(encoder.trainable_variables)
        if train_decoder:
            variables.extend(decoder.trainable_variables)
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

    batch_loss = 0
    decoder_input = tf.constant(start_token, shape=[batch_size])
    decoder_state = decoder.make_initial_state(encoder_state, encoder_output)
    # step_losses = []
    for t in range(max_len):
        decoder_out, decoder_state = decoder(decoder_input,
                                             decoder_state,
                                             encoder_output)
        decoder_input = output_seq[:, t]
        batch_loss += loss_function(decoder_input, decoder_out)
    #     step_loss = loss_function(decoder_input, decoder_out)
    #     step_losses.append(step_loss)
    # step_losses = tf.stack(step_losses, axis=1)
    # batch_loss = tf.reduce_sum(step_losses, axis=1)
    # lengths = tf.cast(batch['length_{}'.format(to_script)], tf.float32)
    # batch_loss = batch_loss / lengths
    # batch_loss = tf.reduce_mean(batch_loss)
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
                            setup,
                            from_script,
                            to_script,
                            patience=3):
    model_setup.pre_training_freeze(setup)
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            strings = decode.extract_strings_from_dataset(valid_data,
                                                          from_script,
                                                          to_script)
            best_val_acc = None
            early_stop_checkpoint = None
            checkpoint_obj = None

            e = 0
            best_e = 0
            while e - best_e <= patience:
                e += 1
                train_encoder = (setup['trainable']['encoder'] is not False
                                 and e >= setup['trainable']['encoder'])
                train_decoder = (setup['trainable']['decoder'] is not False
                                 and e >= setup['trainable']['decoder'])
                loss = run_one_epoch(train_data,
                                     True,
                                     train_encoder=train_encoder,
                                     train_decoder=train_decoder,
                                     from_script=from_script,
                                     to_script=to_script,
                                     encoder=setup['encoder'],
                                     decoder=setup['decoder'],
                                     optimizer=setup['optimizer'],
                                     loss_function=setup['loss_function'])
                val_loss = run_one_epoch(valid_data,
                                         False,
                                         from_script=from_script,
                                         to_script=to_script,
                                         encoder=setup['encoder'],
                                         decoder=setup['decoder'],
                                         loss_function=setup['loss_function'])
                predicted = decode.transliterate(input_strs=strings[from_script],
                                                 from_script=from_script,
                                                 to_script=to_script,
                                                 encoder=setup['encoder'],
                                                 decoder=setup['decoder'],
                                                 decoding_method=decode.beam_search_decode,
                                                 num_beams=10,
                                                 k_best=5)
                acc_at_1 = evaluate.top_k_accuracy(strings[to_script],
                                                   predicted,
                                                   k=1)
                acc_at_5 = evaluate.top_k_accuracy(strings[to_script],
                                                   predicted,
                                                   k=5)
                if checkpoint_obj is None:
                    # have to do so after the graph is built up by using the models
                    checkpoint_obj = setup['make_checkpoint_obj']()
                if best_val_acc is None or acc_at_1 > best_val_acc:
                    best_e = e
                    best_val_acc = acc_at_1
                    early_stop_checkpoint = checkpoint_obj.save(file_prefix='{}/{}'
                                                                .format(tempdir, e))
                print("Epoch {}: Train Loss {:.3f}, Valid Loss {:.3f}, acc@1: {:.3f}, acc@5: {:.3f}"
                      .format(e, loss, val_loss, acc_at_1, acc_at_5))
        except KeyboardInterrupt:
            print('Stopping training because of keyboard interrupt')
        checkpoint_obj.restore(early_stop_checkpoint).assert_consumed()
    print('Validation acc@1: {:.3f}'.format(best_val_acc))
    return best_val_acc


def round_robin_training_regimen(*,
                                 model_setups,
                                 data_pairs,
                                 goal_script,
                                 patience=3):
    for setup in model_setups:
        model_setup.pre_training_freeze(setup)
    patience *= len(model_setups)
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            all_strings = []
            for (train_data, valid_data), setup in zip(data_pairs, model_setups):
                strings = decode.extract_strings_from_dataset(valid_data,
                                                              setup['from_script'],
                                                              setup['to_script'])
                all_strings.append(strings)

            best_val_acc = None
            early_stop_checkpoint = None
            checkpoint_obj = None

            e = 0
            best_e = 0
            while e - best_e <= patience:
                e += 1
                idx = e % len(model_setups)
                setup = model_setups[idx]
                strings = all_strings[idx]
                train_data, valid_data = data_pairs[idx]
                train_encoder = setup['trainable']['encoder'] is not False
                train_decoder = setup['trainable']['decoder'] is not False
                to_script = setup['to_script']
                from_script = setup['from_script']
                loss = run_one_epoch(train_data,
                                     True,
                                     train_encoder=train_encoder,
                                     train_decoder=train_decoder,
                                     from_script=from_script,
                                     to_script=to_script,
                                     encoder=setup['encoder'],
                                     decoder=setup['decoder'],
                                     optimizer=setup['optimizer'],
                                     loss_function=setup['loss_function'])
                val_loss = run_one_epoch(valid_data,
                                         False,
                                         from_script=from_script,
                                         to_script=to_script,
                                         encoder=setup['encoder'],
                                         decoder=setup['decoder'],
                                         loss_function=setup['loss_function'])
                predicted = decode.transliterate(input_strs=strings[from_script],
                                                 from_script=from_script,
                                                 to_script=to_script,
                                                 encoder=setup['encoder'],
                                                 decoder=setup['decoder'],
                                                 decoding_method=decode.beam_search_decode,
                                                 num_beams=10,
                                                 k_best=5)
                acc_at_1 = evaluate.top_k_accuracy(strings[to_script],
                                                   predicted,
                                                   k=1)
                acc_at_5 = evaluate.top_k_accuracy(strings[to_script],
                                                   predicted,
                                                   k=5)
                print('To: {to_script}, '
                      'Epoch {e}: '
                      'Train Loss {train_loss:.3f}, '
                      'Valid Loss {valid_loss:.3f}, '
                      'acc@1: {acc_at_1:.3f}, '
                      'acc@5: {acc_at_5:.3f}'
                      .format(to_script=to_script,
                              e=e,
                              train_loss=loss,
                              valid_loss=val_loss,
                              acc_at_1=acc_at_1,
                              acc_at_5=acc_at_5))
                if e < len(model_setups):
                    # Don't checkpoint until we've trained at least one round
                    continue
                if checkpoint_obj is None:
                    checkpoint_obj = setup['make_checkpoint_obj']()
                if (to_script == goal_script
                    and (best_val_acc is None
                         or acc_at_1 > best_val_acc)):
                    best_e = e
                    best_val_acc = acc_at_1
                    early_stop_checkpoint = checkpoint_obj.save(file_prefix='{}/{}'
                                                                .format(tempdir, e))
        except KeyboardInterrupt:
            print('Stopping training because of keyboard interrupt')
        checkpoint_obj.restore(early_stop_checkpoint).assert_consumed()
        print('Validation acc@1: {:.3f}'.format(best_val_acc))
        return best_val_acc
