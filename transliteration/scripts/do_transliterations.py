import tensorflow as tf

import argparse
import dill as pickle
import json
import sys

from transliteration import model_setup, decode, script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from-script', required=True)
    parser.add_argument('--to-script', required=True)
    parser.add_argument('--encoder-config', required=True)
    parser.add_argument('--decoder-config', required=True)
    parser.add_argument('--save-path', required=True)
    args = parser.parse_args()

    encoder_config = json.loads(args.encoder_config)
    decoder_config = json.loads(args.decoder_config)

    models = model_setup.normal_setup(encoder_config=encoder_config,
                                      decoder_config=decoder_config,
                                      from_script=args.from_script,
                                      to_script=args.to_script)
    saved_weights = pickle.load(open(args.save_path, mode='rb'))
    for m in {'encoder', 'decoder'}:
        models[m].set_weights(saved_weights[m])

    print('Enter strings:')

    for line in sys.stdin:
        line = line.strip()
        results = decode.transliterate(input_strs=[line],
                                       from_script=args.from_script,
                                       to_script=args.to_script,
                                       encoder=models['encoder'],
                                       decoder=models['decoder'],
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
