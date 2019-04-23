import tensorflow as tf

import dill as pickle

from . import model_one, script, train


def normal_setup(*,
                 encoder_config,
                 decoder_config,
                 from_script,
                 to_script):
    optimizer = tf.train.AdamOptimizer()

    config_keys = {**encoder_config,
                   **{'vocab_size': script.SCRIPTS[from_script].vocab_size}}
    encoder_config_obj = model_one.Config(**config_keys)
    config_keys = {**decoder_config,
                   **{'vocab_size': script.SCRIPTS[to_script].vocab_size}}
    decoder_config_obj = model_one.Config(**config_keys)

    encoder = model_one.Encoder(encoder_config_obj)
    decoder = model_one.Decoder(decoder_config_obj)

    # get at least the main variables set up
    train.exercise_encoder_decoder(encoder=encoder,
                                   decoder=decoder,
                                   to_script=to_script)

    def make_checkpoint_obj():
        return tf.train.Checkpoint(optimizer=optimizer,
                                   encoder=encoder,
                                   decoder=decoder)

    return {'setup': 'normal',
            'from_script': from_script,
            'to_script': to_script,
            'optimizer': optimizer,
            'encoder': encoder,
            'decoder': decoder,
            'config': {'encoder': encoder_config, 'decoder': decoder_config},
            'trainable': {'encoder': True, 'decoder': True},
            'make_checkpoint_obj': make_checkpoint_obj,
            'loss_function': model_one.loss_function}


def transfer_learning_setup(*,
                            encoder_config,
                            decoder_config,
                            transfer_encoder_config,
                            transfer_decoder_config,
                            from_script,
                            transfer_to_script,
                            to_script,
                            style='normal',
                            restore=False):
    """First train on from_script -> transfer_to_script, then transfer the encoder to
    from_script -> to_script.

    """

    optimizer = tf.train.AdamOptimizer()

    if style == 'normal' or style == 'default':
        transfer_setup = _normal_transfer_setup
    elif style == 'stacked':
        transfer_setup = _stacked_transfer_setup
    elif style == 'combined':
        transfer_setup = _combined_transfer_setup

    initial, main =\
        transfer_setup(from_script=from_script,
                       encoder_config=encoder_config,
                       decoder_config=decoder_config,
                       transfer_encoder_config=transfer_encoder_config,
                       transfer_decoder_config=transfer_decoder_config,
                       optimizer=optimizer,
                       to_script=to_script,
                       transfer_to_script=transfer_to_script,
                       restore=restore)

    for setup in [initial, main]:
        if setup['encoder'] is None or setup['decoder'] is None:
            # We have a partial setup when restoring
            continue
        train.exercise_encoder_decoder(encoder=setup['encoder'],
                                       decoder=setup['decoder'],
                                       to_script=setup['to_script'])
        setup['loss_function'] = model_one.loss_function
        setup['optimizer'] = optimizer

    return initial, main


def _normal_transfer_setup(*,
                           from_script,
                           encoder_config,
                           decoder_config,
                           transfer_encoder_config,
                           transfer_decoder_config,
                           to_script,
                           optimizer,
                           transfer_to_script,
                           restore):
    config_keys = {**transfer_decoder_config,
                   **{'vocab_size': script.SCRIPTS[transfer_to_script].vocab_size}}
    t_decoder_config_obj = model_one.Config(**config_keys)
    transfer_decoder = model_one.Decoder(t_decoder_config_obj)

    config_keys = {**encoder_config,
                   **{'vocab_size': script.SCRIPTS[from_script].vocab_size}}
    encoder_config_obj = model_one.Config(**config_keys)
    config_keys = {**decoder_config,
                   **{'vocab_size': script.SCRIPTS[to_script].vocab_size}}
    decoder_config_obj = model_one.Config(**config_keys)
    encoder = model_one.Encoder(encoder_config_obj)
    decoder = model_one.Decoder(decoder_config_obj)

    def make_checkpoint_obj():
        return tf.train.Checkpoint(optimizer=optimizer,
                                   encoder=encoder,
                                   transfer_decoder=transfer_decoder,
                                   decoder=decoder)

    initial = {'setup': 'normal',
               'encoder': encoder,
               'decoder': transfer_decoder,
               'trainable': {'encoder': True, 'decoder': True},
               'from_script': from_script,
               'to_script': transfer_to_script,
               'make_checkpoint_obj': make_checkpoint_obj,
               'config': {'encoder': encoder_config,
                          'decoder': transfer_decoder_config}}
    main = {'setup': 'normal',
            'encoder': encoder,
            'decoder': decoder,
            'trainable': {'encoder': 3, 'decoder': True},
            'from_script': from_script,
            'to_script': to_script,
            'make_checkpoint_obj': make_checkpoint_obj,
            'config': {'encoder': encoder_config,
                       'decoder': decoder_config}}
    return initial, main


def _combined_transfer_setup(*,
                             from_script,
                             encoder_config,
                             decoder_config,
                             transfer_encoder_config,
                             transfer_decoder_config,
                             to_script,
                             optimizer,
                             transfer_to_script,
                             restore):

    config_keys = {**transfer_encoder_config,
                   **{'vocab_size': script.SCRIPTS[from_script].vocab_size}}
    encoder_config_obj = model_one.Config(**config_keys)
    transfer_encoder = model_one.Encoder(encoder_config_obj)

    config_keys = {**decoder_config,
                   **{'vocab_size': script.SCRIPTS[to_script].vocab_size}}
    decoder_config_obj = model_one.Config(**config_keys)
    decoder = model_one.Decoder(decoder_config_obj)
    config_keys = {**encoder_config,
                   **{'vocab_size': script.SCRIPTS[from_script].vocab_size}}
    encoder_config_obj = model_one.Config(**config_keys)
    encoder = model_one.CombinedEncoder(encoder_config_obj, transfer_encoder)

    transfer_decoder = None
    if not restore:
        config_keys = {**transfer_decoder_config,
                       **{'vocab_size': script.SCRIPTS[transfer_to_script].vocab_size}}
        t_decoder_config_obj = model_one.Config(**config_keys)
        transfer_decoder = model_one.Decoder(t_decoder_config_obj)
    optional = not restore and {'transfer_decoder': transfer_decoder} or dict()

    def make_checkpoint_obj():
        return tf.train.Checkpoint(optimizer=optimizer,
                                   transfer_encoder=transfer_encoder,
                                   encoder=encoder,
                                   decoder=decoder,
                                   **optional)

    initial = {'setup': 'normal',
               'encoder': transfer_encoder,
               'decoder': transfer_decoder,
               'trainable': {'encoder': True, 'decoder': True},
               'from_script': from_script,
               'to_script': transfer_to_script,
               'make_checkpoint_obj': make_checkpoint_obj,
               'config': {'encoder': transfer_encoder_config,
                          'decoder': transfer_decoder_config}}
    main = {'setup': 'combined',
            'encoder': encoder,
            'decoder': decoder,
            'transfer_encoder': transfer_encoder,
            'trainable': {'encoder': True, 'decoder': True},
            'freeze': ['transfer_encoder'],
            'from_script': from_script,
            'to_script': to_script,
            'make_checkpoint_obj': make_checkpoint_obj,
            'config': {'encoder': encoder_config,
                       'decoder': decoder_config,
                       'transfer_encoder': transfer_encoder_config,
                       'transfer_decoder': transfer_decoder_config}}
    return initial, main


def _stacked_transfer_setup(*,
                            from_script,
                            transfer_encoder_config,
                            transfer_decoder_config,
                            encoder_config,
                            decoder_config,
                            to_script,
                            optimizer,
                            transfer_to_script,
                            restore):
    config_keys = {**transfer_encoder_config,
                   **{'vocab_size': script.SCRIPTS[from_script].vocab_size}}
    t_encoder_config_obj = model_one.Config(**config_keys)
    transfer_encoder = model_one.Encoder(t_encoder_config_obj)
    config_keys = {**transfer_decoder_config,
                   **{'vocab_size': script.SCRIPTS[transfer_to_script].vocab_size}}
    t_decoder_config_obj = model_one.Config(**config_keys)
    transfer_decoder = model_one.Decoder(t_decoder_config_obj)
    encoder = model_one.StackedEncoderDecoderEncoder(transfer_encoder,
                                                     transfer_decoder,
                                                     transfer_to_script)
    config_keys = {**decoder_config,
                   **{'vocab_size': script.SCRIPTS[to_script].vocab_size}}
    decoder_config_obj = model_one.Config(**config_keys)
    decoder = model_one.Decoder(decoder_config_obj)

    def make_checkpoint_obj():
        return tf.train.Checkpoint(optimizer=optimizer,
                                   transfer_encoder=transfer_encoder,
                                   transfer_decoder=transfer_decoder,
                                   encoder=encoder,
                                   decoder=decoder)

    initial = {'setup': 'normal',
               'encoder': transfer_encoder,
               'decoder': transfer_decoder,
               'trainable': {'encoder': True, 'decoder': True},
               'from_script': from_script,
               'to_script': transfer_to_script,
               'make_checkpoint_obj': make_checkpoint_obj,
               'config': {'encoder': transfer_encoder_config,
                          'decoder': transfer_decoder_config}}
    main = {'setup': 'stacked',
            'encoder': encoder,
            'decoder': decoder,
            'trainable': {'encoder': False, 'decoder': True},
            # 'transfer_encoder': transfer_encoder,
            # 'transfer_decoder': transfer_decoder,
            'from_script': from_script,
            'to_script': to_script,
            'transfer_to_script': transfer_to_script,
            'make_checkpoint_obj': make_checkpoint_obj,
            'config': {'encoder': encoder_config,
                       'decoder': decoder_config,
                       'transfer_encoder': transfer_encoder_config,
                       'transfer_decoder': transfer_decoder_config}}
    return initial, main


def save_to_pkl(setup, save_path):
    additional_dump = dict()
    setup_style = setup['setup']
    saved_models = {'encoder', 'decoder'}
    if setup_style in {'normal', 'combined'}:
        pass
    elif setup_style == 'stacked':
        # saved_models = {'encoder', 'decoder', 'transfer_encoder', 'transfer_decoder'}
        additional_dump['transfer_to_script'] = setup['transfer_to_script']
    else:
        raise ValueError()

    saved_weights = {m: setup[m].get_weights()
                     for m in saved_models}
    to_dump = {**{'setup': setup['setup'],
                  'weights': saved_weights,
                  'config': setup['config'],
                  'from_script': setup['from_script'],
                  'to_script': setup['to_script']},
               **additional_dump}
    pickle.dump(to_dump, open(save_path, mode='wb'))


def load_from_pkl(save_path):
    dump = pickle.load(open(save_path, mode='rb'))
    setup_style = dump['setup']
    saved_models = {'encoder', 'decoder'}
    if setup_style == 'normal':
        result = normal_setup(encoder_config=dump['config']['encoder'],
                              decoder_config=dump['config']['decoder'],
                              from_script=dump['from_script'],
                              to_script=dump['to_script'])
    elif setup_style in {'stacked', 'combined'}:
        # these two things can be None
        transfer_decoder_config = dump['config'].get('transfer_decoder')
        transfer_to_script = dump.get('transfer_to_script')
        _, result = transfer_learning_setup(encoder_config=dump['config']['encoder'],
                                            decoder_config=dump['config']['decoder'],
                                            transfer_encoder_config=dump['config']['transfer_encoder'],
                                            transfer_decoder_config=transfer_decoder_config,
                                            from_script=dump['from_script'],
                                            to_script=dump['to_script'],
                                            transfer_to_script=transfer_to_script,
                                            style=setup_style,
                                            restore=True)
    else:
        raise ValueError()
    saved_weights = dump['weights']
    for m in saved_models:
        result[m].set_weights(saved_weights[m])
    return result


def pre_training_freeze(setup):
    if 'freeze' in setup:
        for m in setup['freeze']:
            m = setup[m]
            for l in m.layers:
                l.trainable = False
