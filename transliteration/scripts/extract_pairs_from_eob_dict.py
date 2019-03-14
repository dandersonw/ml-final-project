import pandas as pd

import argparse
import unicodedata

from transliteration import script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('eob_dict')
    parser.add_argument('--output-file',
                        default='/dev/stdout')
    args = parser.parse_args()

    english_script = script.SCRIPTS['en']
    katakana_script = script.SCRIPTS['ja']

    eob_dict = pd.read_csv(args.eob_dict,
                           names=['en', 'ja', 'romanized_ja'],
                           header=None,
                           keep_default_na=False)
    eob_dict = eob_dict[eob_dict['ja']\
                        .map(lambda s: all(katakana_script._char_in_range(c)
                                           for c in katakana_script.preprocess_string(s)))]
    eob_dict = eob_dict[eob_dict['en']\
                        .map(lambda s: all(english_script._char_in_range(c)
                                           for c in english_script.preprocess_string(s)))]
    eob_dict.to_csv(args.output_file, columns=['en', 'ja'], index=False)


if __name__ == '__main__':
    main()
