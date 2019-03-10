import pandas as pd

import argparse
import unicodedata

from transliteration import data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('muse_dict')
    parser.add_argument('--output-file',
                        default='/dev/stdout')
    parser.add_argument('--direction',
                        default='en_ja',
                        choices=['en_ja', 'ja_en'])
    args = parser.parse_args()

    muse_dict = pd.read_csv(args.muse_dict,
                            sep='\t',
                            names=args.direction.split('_'),
                            keep_default_na=False)
    muse_dict['ja'] = muse_dict['ja'].map(lambda s: unicodedata.normalize('NFKC', s))
    muse_dict = muse_dict[muse_dict['ja'].map(lambda s: is_katakana(s))]
    muse_dict.to_csv(args.output_file, index=False)


def is_katakana(chars):
    return all(ord(c) >= data.KATAKANA_BLOCK_START
               and ord(c) <= data.KATAKANA_BLOCK_END
               for c in chars)


if __name__ == '__main__':
    main()
