import tensorflow as tf
import numpy as np

import argparse
import dill as pickle
import json
from itertools import chain

from transliteration import model_setup, train, data, decode, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data',
                        help='Test data in TFRecord format',
                        required=True)
    parser.add_argument('--data-kwargs', default='{}')
    parser.add_argument('--from-script', required=True)
    parser.add_argument('--to-script', required=True)
    parser.add_argument('--encoder-config', required=True)
    parser.add_argument('--decoder-config', required=True)
    parser.add_argument('--save-path', required=True)
    args = parser.parse_args()

    encoder_config = json.loads(args.encoder_config)
    decoder_config = json.loads(args.decoder_config)
    data_kwargs = json.loads(args.data_kwargs)

    models = model_setup.normal_setup(encoder_config=encoder_config,
                                      decoder_config=decoder_config,
                                      from_script=args.from_script,
                                      to_script=args.to_script)
    saved_weights = pickle.load(open(args.save_path, mode='rb'))
    for m in {'encoder', 'decoder'}:
        models[m].set_weights(saved_weights[m])

    test_data = data.make_dataset(args.test_data,
                                  from_script=args.from_script,
                                  to_script=args.to_script,
                                  batch_size=32,
                                  **data_kwargs)

    from_strings = []
    to_strings = []
    for batch in test_data:
        from_tokens = np.expand_dims(batch[args.from_script], 1)
        to_tokens = np.expand_dims(batch[args.to_script], 1)
        from_strings.append(decode.deintern_decode_results(from_tokens, args.from_script))
        to_strings.append(decode.deintern_decode_results(to_tokens, args.to_script))

    from_strings = list(chain.from_iterable(chain.from_iterable(from_strings)))
    to_strings = list(chain.from_iterable(chain.from_iterable(to_strings)))

    predicted = decode.transliterate(input_strs=from_strings,
                                     from_script=args.from_script,
                                     to_script=args.to_script,
                                     encoder=models['encoder'],
                                     decoder=models['decoder'],
                                     decoding_method=decode.beam_search_decode,
                                     num_beams=10,
                                     k_best=5)

    acc_at_1 = evaluate.top_k_accuracy(to_strings, predicted, k=1)
    acc_at_5 = evaluate.top_k_accuracy(to_strings, predicted, k=5)
    mrr_at_5 = evaluate.mrr(to_strings, predicted, k=5)
    print('Accuracy at k=1: {:.3f}'.format(acc_at_1))
    print('Accuracy at k=5: {:.3f}'.format(acc_at_5))
    print('MRR at k=5: {:.3f}'.format(mrr_at_5))

    loss = train.run_one_epoch(test_data,
                               False,
                               from_script=args.from_script,
                               to_script=args.to_script,
                               encoder=models['encoder'],
                               decoder=models['decoder'],
                               loss_function=models['loss_function'])

    print('Test Loss: {:.3f}'.format(loss))


if __name__ == '__main__':
    tf.logging.set_verbosity('ERROR')
    tf.enable_eager_execution()
    main()
