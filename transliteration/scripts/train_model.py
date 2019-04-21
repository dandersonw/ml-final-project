import tensorflow as tf

import argparse
import json

from transliteration import model_setup, train, data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data',
                        help='Train data in TFRecord format',
                        required=True)
    parser.add_argument('--valid-data',
                        help='Validation data in TFRecord format',
                        required=True)
    parser.add_argument('--transfer-train-data',
                        help='Train data in TFRecord format')
    parser.add_argument('--transfer-valid-data',
                        help='Validation data in TFRecord format')
    parser.add_argument('--data-kwargs', default='{}')
    parser.add_argument('--transfer-data-kwargs', default='{}')
    parser.add_argument('--from-script', required=True)
    parser.add_argument('--to-script', required=True)
    parser.add_argument('--encoder-config', required=True)
    parser.add_argument('--decoder-config', required=True)
    parser.add_argument('--start-save')
    parser.add_argument('--transfer-to-script')
    parser.add_argument('--transfer-encoder-config')
    parser.add_argument('--transfer-decoder-config')
    parser.add_argument('--transfer-start-save')
    parser.add_argument('--transfer-style', default='normal')
    parser.add_argument('--transfer-training', default='sequential')
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--random-restarts', default=3, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()

    encoder_config = json.loads(args.encoder_config)
    transfer_encoder_config = (encoder_config
                               if args.transfer_encoder_config is None
                               else json.loads(args.transfer_encoder_config))
    decoder_config = json.loads(args.decoder_config)
    transfer_decoder_config = (decoder_config
                               if args.transfer_decoder_config is None
                               else json.loads(args.transfer_decoder_config))
    data_kwargs = json.loads(args.data_kwargs)
    transfer_data_kwargs = json.loads(args.transfer_data_kwargs)

    best_val_acc = None

    for r in range(args.random_restarts):
        print('Random restart No. {}'.format(r))
        setup, val_acc = train_model(encoder_config=encoder_config,
                                     decoder_config=decoder_config,
                                     data_kwargs=data_kwargs,
                                     transfer_data_kwargs=transfer_data_kwargs,
                                     train_data=args.train_data,
                                     valid_data=args.valid_data,
                                     start_save=args.start_save,
                                     transfer_encoder_config=transfer_encoder_config,
                                     transfer_decoder_config=transfer_decoder_config,
                                     transfer_train_data=args.transfer_train_data,
                                     transfer_valid_data=args.transfer_valid_data,
                                     transfer_start_save=args.transfer_start_save,
                                     transfer_style=args.transfer_style,
                                     transfer_training=args.transfer_training,
                                     from_script=args.from_script,
                                     to_script=args.to_script,
                                     transfer_to_script=args.transfer_to_script,
                                     batch_size=args.batch_size)
        if best_val_acc is None or val_acc > best_val_acc:
            model_setup.save_to_pkl(setup, args.save_path)
            best_val_acc = val_acc
            print('Best validation acc: {:.3f}'.format(best_val_acc))


def train_model(*,
                encoder_config,
                decoder_config,
                data_kwargs,
                transfer_data_kwargs,
                train_data,
                valid_data,
                start_save=None,
                transfer_style=None,
                transfer_train_data=None,
                transfer_valid_data=None,
                transfer_encoder_config=None,
                transfer_decoder_config=None,
                transfer_start_save=None,
                transfer_training=None,
                from_script,
                to_script,
                transfer_to_script=None,
                batch_size):

    doing_transfer_learning = (transfer_train_data is not None
                               or transfer_valid_data is not None
                               or transfer_start_save is not None)
    if doing_transfer_learning:
        assert (transfer_train_data is not None
                and transfer_style is not None
                and transfer_valid_data is not None
                and transfer_to_script is not None
                and transfer_encoder_config is not None
                and transfer_decoder_config is not None
                or transfer_start_save is not None)

        if transfer_start_save is not None:
            # TODO - check for config consistency?
            saved_setup = model_setup.load_from_pkl(transfer_start_save)
            transfer_to_script = saved_setup['to_script']
            transfer_encoder_config = saved_setup['config']['encoder']
            transfer_decoder_config = saved_setup['config']['decoder']

        initial_setup, setup = \
            model_setup.transfer_learning_setup(encoder_config=encoder_config,
                                                decoder_config=decoder_config,
                                                transfer_encoder_config=transfer_encoder_config,
                                                transfer_decoder_config=transfer_decoder_config,
                                                style=transfer_style,
                                                from_script=from_script,
                                                transfer_to_script=transfer_to_script,
                                                to_script=to_script)

        if transfer_start_save is not None:
            for m in {'encoder', 'decoder'}:
                # TODO - hand off to model_setup?
                initial_setup[m].set_weights(saved_setup[m].get_weights())
            del saved_setup
        else:
            transfer_train_data = data.make_dataset(transfer_train_data,
                                                    from_script=from_script,
                                                    to_script=transfer_to_script,
                                                    batch_size=batch_size,
                                                    **transfer_data_kwargs)
            transfer_valid_data = data.make_dataset(transfer_valid_data,
                                                    from_script=from_script,
                                                    to_script=transfer_to_script,
                                                    batch_size=batch_size,
                                                    **transfer_data_kwargs)
    else:
        setup = model_setup.normal_setup(encoder_config=encoder_config,
                                         decoder_config=decoder_config,
                                         from_script=from_script,
                                         to_script=to_script)

    train_data = data.make_dataset(train_data,
                                   from_script=from_script,
                                   to_script=to_script,
                                   batch_size=batch_size,
                                   **data_kwargs)
    valid_data = data.make_dataset(valid_data,
                                   from_script=from_script,
                                   to_script=to_script,
                                   batch_size=batch_size,
                                   **data_kwargs)

    if start_save is not None:
        # TODO - load config from the pickle?
        saved_setup = model_setup.load_from_pkl(start_save)
        for m in {'encoder', 'decoder'}:
            # TODO - hand off to model_setup?
            setup[m].set_weights(saved_setup[m].get_weights())
        del saved_setup

    if doing_transfer_learning:
        if transfer_training == 'sequential' and transfer_start_save is None:
            train.normal_training_regimen(train_data=transfer_train_data,
                                          valid_data=transfer_valid_data,
                                          from_script=from_script,
                                          to_script=transfer_to_script,
                                          setup=initial_setup)
        if transfer_training == 'sequential':
            final_val_acc = train.normal_training_regimen(train_data=train_data,
                                                          valid_data=valid_data,
                                                          from_script=from_script,
                                                          to_script=to_script,
                                                          setup=setup)
        elif transfer_training == 'round-robin':
            setups = [initial_setup, setup]
            data_pairs = [(transfer_train_data, transfer_valid_data),
                          (train_data, valid_data)]
            final_val_acc = train.round_robin_training_regimen(model_setups=setups,
                                                               data_pairs=data_pairs,
                                                               goal_script=to_script)
    else:
        final_val_acc = train.normal_training_regimen(train_data=train_data,
                                                      valid_data=valid_data,
                                                      from_script=from_script,
                                                      to_script=to_script,
                                                      setup=setup)
    return setup, final_val_acc


if __name__ == '__main__':
    tf.logging.set_verbosity('ERROR')
    tf.enable_eager_execution()
    main()
