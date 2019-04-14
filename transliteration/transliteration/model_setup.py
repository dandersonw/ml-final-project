import tensorflow as tf

import dill as pickle

from . import model_one, script, train


def normal_setup(*,
                 encoder_config,
                 decoder_config,
                 from_script,
                 to_script):
    optimizer = tf.train.AdamOptimizer()

    encoder_config = {**encoder_config,
                      **{'vocab_size': script.SCRIPTS[from_script].vocab_size}}
    encoder_config = model_one.Config(**encoder_config)
    decoder_config = {**decoder_config,
                      **{'vocab_size': script.SCRIPTS[to_script].vocab_size}}
    decoder_config = model_one.Config(**decoder_config)

    encoder = model_one.Encoder(encoder_config)
    decoder = model_one.Decoder(decoder_config)

    # get at least the main variables set up
    train.exercise_encoder_decoder(encoder=encoder,
                                   decoder=decoder,
                                   to_script=to_script)

    def make_checkpoint_obj():
        return tf.train.Checkpoint(optimizer=optimizer,
                                   encoder=encoder,
                                   decoder=decoder)

    return {'optimizer': optimizer,
            'encoder': encoder,
            'decoder': decoder,
            'make_checkpoint_obj': make_checkpoint_obj,
            'loss_function': model_one.loss_function}


def transfer_learning_setup(*,
                            encoder_config,
                            decoder_config,
                            from_script,
                            transfer_to_script,
                            to_script):
    """First train on from_script -> transfer_to_script, then transfer the encoder to
    from_script -> to_script.

    """

    optimizer = tf.train.AdamOptimizer()

    encoder_config = {**encoder_config,
                      **{'vocab_size': script.SCRIPTS[from_script].vocab_size}}
    encoder_config = model_one.Config(**encoder_config)
    decoder_config_initial = {**decoder_config,
                              **{'vocab_size': script.SCRIPTS[transfer_to_script].vocab_size}}
    decoder_config_initial = model_one.Config(**decoder_config_initial)
    decoder_config = {**decoder_config,
                      **{'vocab_size': script.SCRIPTS[to_script].vocab_size}}
    decoder_config = model_one.Config(**decoder_config)

    encoder = model_one.Encoder(encoder_config)
    decoder_initial = model_one.Decoder(decoder_config)
    decoder = model_one.Decoder(decoder_config)

    train.exercise_encoder_decoder(encoder=encoder,
                                   decoder=decoder_initial,
                                   to_script=transfer_to_script)
    train.exercise_encoder_decoder(encoder=encoder,
                                   decoder=decoder,
                                   to_script=to_script)

    def make_checkpoint_obj():
        # we will want to rerun this when the graph changes,
        # so make it a thunk
        return tf.train.Checkpoint(optimizer=optimizer,
                                   encoder=encoder,
                                   decoder_initial=decoder_initial,
                                   decoder=decoder)

    return {'optimizer': optimizer,
            'encoder': encoder,
            'decoder_initial': decoder_initial,
            'decoder': decoder,
            'make_checkpoint_obj': make_checkpoint_obj,
            'loss_function': model_one.loss_function}


def save_weights_to_pkl(models, save_path):
    saved_weights = {m: models[m].get_weights()
                     for m in {'encoder', 'decoder'}}
    pickle.dump(saved_weights, open(save_path, mode='wb'))


def load_weights_from_pkl(models, save_path):
    saved_weights = pickle.load(open(save_path, mode='rb'))
    for m in {'encoder', 'decoder'}:
        models[m].set_weights(saved_weights[m])
