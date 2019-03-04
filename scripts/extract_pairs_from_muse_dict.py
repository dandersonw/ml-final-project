import pandas as pd

import argparse
import regex as re
import unicodedata


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
    katakana_re = re.compile(r'\p{IsKatakana}')
    muse_dict = muse_dict[muse_dict['ja'].map(lambda s: katakana_re.search(s) is not None)]
    muse_dict.to_csv(args.output_file, index=False)


if __name__ == '__main__':
    main()
