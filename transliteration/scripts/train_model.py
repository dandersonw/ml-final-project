import tensorflow as tf

import argparse
import json
import dill as pickle

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
    parser.add_argument('--transfer-to-script')
    parser.add_argument('--encoder-config', required=True)
    parser.add_argument('--decoder-config', required=True)
    parser.add_argument('--save-path', required=True)
    parser.add_argument('--random-restarts', default=3, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    args = parser.parse_args()

    encoder_config = json.loads(args.encoder_config)
    decoder_config = json.loads(args.decoder_config)
    data_kwargs = json.loads(args.data_kwargs)
    transfer_data_kwargs = json.loads(args.transfer_data_kwargs)

    best_val_loss = None

    for r in range(args.random_restarts):
        print('Random restart No. {}'.format(r))
        models, val_loss = train_model(encoder_config=encoder_config,
                                       decoder_config=decoder_config,
                                       data_kwargs=data_kwargs,
                                       transfer_data_kwargs=transfer_data_kwargs,
                                       train_data=args.train_data,
                                       valid_data=args.valid_data,
                                       transfer_train_data=args.transfer_train_data,
                                       transfer_valid_data=args.transfer_valid_data,
                                       from_script=args.from_script,
                                       to_script=args.to_script,
                                       transfer_to_script=args.transfer_to_script,
                                       batch_size=args.batch_size)
        if best_val_loss is None or val_loss < best_val_loss:
            model_setup.save_weights_to_pkl(models, args.save_path)
            best_val_loss = val_loss
    print('Best validation loss: {:.3f}'.format(best_val_loss))


def train_model(*,
                encoder_config,
                decoder_config,
                data_kwargs,
                transfer_data_kwargs,
                train_data,
                valid_data,
                transfer_train_data=None,
                transfer_valid_data=None,
                from_script,
                to_script,
                transfer_to_script=None,
                batch_size):

    doing_transfer_learning = (transfer_train_data is not None
                               or transfer_valid_data is not None)
    if doing_transfer_learning:
        assert (transfer_valid_data is not None
                and transfer_train_data is not None
                and transfer_to_script is not None)
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
        models = model_setup.transfer_learning_setup(encoder_config=encoder_config,
                                                     decoder_config=decoder_config,
                                                     from_script=from_script,
                                                     transfer_to_script=transfer_to_script,
                                                     to_script=transfer_to_script)
        models_ = {**models, **{'decoder': models['decoder_initial']}}
        train.normal_training_regimen(train_data=transfer_train_data,
                                      valid_data=transfer_valid_data,
                                      from_script=from_script,
                                      to_script=transfer_to_script,
                                      models=models_)
    else:
        models = model_setup.normal_setup(encoder_config=encoder_config,
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

    final_val_loss = train.normal_training_regimen(train_data=train_data,
                                                   valid_data=valid_data,
                                                   from_script=from_script,
                                                   to_script=to_script,
                                                   models=models)
    return models, final_val_loss


if __name__ == '__main__':
    tf.logging.set_verbosity('ERROR')
    tf.enable_eager_execution()
    main()
