import pandas as pd

import argparse
from collections import Counter, deque
from transliteration import data
from transliteration.script import SCRIPTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('our_translations')
    parser.add_argument('--output-file',
                        default='/dev/stdout')
    parser.add_argument('--direction',
                        default='ja_en',
                        choices=['en_ja', 'ja_en'])
    parser.add_argument('--frequency-cutoff',
                        required=False,
                        type=int)
    parser.add_argument('--noise-commonness-cutoff',
                        default=20,
                        type=int)
    parser.add_argument('--output-k',
                        default=1,
                        type=int)
    args = parser.parse_args()

    translations = pd.read_csv(args.our_translations,
                               keep_default_na=False)

    from_script_name, to_script_name = args.direction.split('_')
    from_script = SCRIPTS[from_script_name]
    to_script = SCRIPTS[to_script_name]

    input_k = 0
    while '{}_{}'.format(to_script_name, input_k + 1) in translations.columns:
        input_k += 1
    assert input_k > 0

    to_columns = ['{}_{}'.format(to_script_name, k) for k in range(1, input_k + 1)]

    count = Counter()
    for c in to_columns:
        count.update(translations[c])
    too_common = {k for k, v in count.items()
                  if v >= args.noise_commonness_cutoff}

    out_from = deque()
    out_to = deque()
    for i, row in translations.iterrows():
        from_token = row[from_script_name]
        if (not valid_raw_str(from_token, from_script)
            or (args.frequency_cutoff is not None
                and row['frequency_rank'] < args.frequency_cutoff)):
            continue

        num_output = 0
        for c in to_columns:
            if num_output == args.output_k:
                break

            to_token = row[c]
            if valid_output_choice(to_token, to_script, too_common):
                num_output += 1
                out_from.append(from_token)
                out_to.append(to_token)

    translations = pd.DataFrame({from_script_name: out_from,
                                 to_script_name: out_to})
    translations.to_csv(args.output_file, index=None)


def valid_raw_str(string, script):
    return all(script._char_in_range(c) for c in script.preprocess_string(string))


def valid_output_choice(string, script, too_common):
    return valid_raw_str(string, script) and string not in too_common


if __name__ == '__main__':
    main()
