import pandas as pd

import argparse
import re

from transliteration import script


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cmu_dict')
    parser.add_argument('--output-file',
                        default='/dev/stdout')
    args = parser.parse_args()
    english_script: script.Script = script.SCRIPTS['en']

    words = []
    pronunciations = []

    with open(args.cmu_dict, mode='r', encoding='latin1') as input_file:
        for line in input_file:
            if line.startswith(';;;'):
                continue

            word = line[:line.index(' ')]
            pronunciation = line[line.index(' '):]

            word = re.sub(r'\([0-9]\)$', '', word)
            pronunciation = pronunciation.strip()
            if not all(english_script._char_in_range(c)
                       for c in english_script.preprocess_string(word)):
                continue

            words.append(word)
            pronunciations.append(pronunciation)

    df = pd.DataFrame({'en': words, 'cmu': pronunciations})
    df.to_csv(args.output_file, index=None)


if __name__ == '__main__':
    main()
