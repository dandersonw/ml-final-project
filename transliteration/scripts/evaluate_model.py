import tensorflow as tf
import numpy as np

import argparse
import json

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

    test_strings = decode.extract_strings_from_dataset(test_data,
                                                       from_script,
                                                       to_script)

    predicted = decode.transliterate(input_strs=test_strings[from_script],
                                     from_script=from_script,
                                     to_script=to_script,
                                     encoder=setup['encoder'],
                                     decoder=setup['decoder'],
                                     decoding_method=decode.beam_search_decode,
                                     num_beams=10,
                                     k_best=5)

    acc_at_1 = evaluate.top_k_accuracy(test_strings[to_script],
                                       predicted,
                                       k=1)
    acc_at_5 = evaluate.top_k_accuracy(test_strings[to_script],
                                       predicted,
                                       k=5)
    mrr_at_5 = evaluate.mrr(test_strings[to_script],
                            predicted,
                            k=5)
    cee = evaluate.character_error_rate(test_strings[to_script],
                                        predicted,
                                        script_name=to_script)
    print('Accuracy at k=1: {:.3f}'.format(acc_at_1))
    print('Accuracy at k=5: {:.3f}'.format(acc_at_5))
    print('MRR at k=5: {:.3f}'.format(mrr_at_5))
    print('Character error rate: {:.3f}'.format(cee))


if __name__ == '__main__':
    tf.logging.set_verbosity('ERROR')
    tf.enable_eager_execution()
    main()
