import tensorflow as tf

import argparse
import sys

from transliteration import model_setup, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', required=True)
    args = parser.parse_args()

    setup = model_setup.load_from_pkl(args.save_path)
    from_script = setup['from_script']
    to_script = setup['to_script']

    print('Enter strings:')

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        results = decode.transliterate(input_strs=[line],
                                       from_script=from_script,
                                       to_script=to_script,
                                       encoder=setup['encoder'],
                                       decoder=setup['decoder'],
                                       decoding_method=decode.beam_search_decode,
                                       num_beams=10,
                                       k_best=5)
        hyps, weights = results
        hyps = hyps[0]
        weights = weights[0]
        print(', '.join(['{}: {:.3f}'.format(h, w) for h, w in zip(hyps, weights)]))


if __name__ == '__main__':
    tf.logging.set_verbosity('ERROR')
    tf.enable_eager_execution()
    main()
