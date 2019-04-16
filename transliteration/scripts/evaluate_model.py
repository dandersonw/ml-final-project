import tensorflow as tf
import numpy as np

import argparse
import json
from itertools import chain

from transliteration import model_setup, train, data, decode, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-data',
                        help='Test data in TFRecord format',
                        required=True)
    parser.add_argument('--data-kwargs', default='{}')
    parser.add_argument('--save-path', required=True)
    args = parser.parse_args()

    data_kwargs = json.loads(args.data_kwargs)

    setup = model_setup.load_from_pkl(args.save_path)
    from_script = setup['from_script']
    to_script = setup['to_script']

    test_data = data.make_dataset(args.test_data,
                                  from_script=from_script,
                                  to_script=to_script,
                                  batch_size=32,
                                  **data_kwargs)

    from_strings = []
    to_strings = []
    for batch in test_data:
        from_tokens = np.expand_dims(batch[from_script], 1)
        to_tokens = np.expand_dims(batch[to_script], 1)
        from_strings.append(decode.deintern_decode_results(from_tokens, from_script))
        to_strings.append(decode.deintern_decode_results(to_tokens, to_script))

    from_strings = list(chain.from_iterable(chain.from_iterable(from_strings)))
    to_strings = list(chain.from_iterable(chain.from_iterable(to_strings)))

    predicted = decode.transliterate(input_strs=from_strings,
                                     from_script=from_script,
                                     to_script=to_script,
                                     encoder=setup['encoder'],
                                     decoder=setup['decoder'],
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
                               from_script=from_script,
                               to_script=to_script,
                               encoder=setup['encoder'],
                               decoder=setup['decoder'],
                               loss_function=setup['loss_function'])

    print('Test Loss: {:.3f}'.format(loss))


if __name__ == '__main__':
    tf.logging.set_verbosity('ERROR')
    tf.enable_eager_execution()
    main()
