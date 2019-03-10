import pandas as pd
import numpy as np

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_prefix', required=True)
    parser.add_argument('--splits',
                        help='In format of name/weight.',
                        nargs='*',
                        required=True)
    parser.add_argument('--input_files', nargs='*', required=True)
    parser.add_argument('--random-seed', default=1001)
    args = parser.parse_args()

    splits = [sp.split('/') for sp in args.splits]
    splits = [(sp[0], int(sp[1])) for sp in splits]
    weight_sum = sum(sp[1] for sp in splits)
    splits = [(sp[0], sp[1] / weight_sum) for sp in splits]

    input_data = pd.concat([pd.read_csv(input_file, keep_default_na=False)
                            for input_file in args.input_files])

    np.random.seed(args.random_seed)
    random_values = np.random.rand(len(input_data))

    results = {sp[0]: [] for sp in splits}
    for rand, example in zip(random_values, input_data.iterrows()):
        _, example = example
        for name, weight in splits:
            if rand < weight:
                results[name].append(example)
                break
            else:
                rand -= weight

    for name, rows in results.items():
        df = pd.concat(rows, axis=1).T
        path = '{}_{}.csv'.format(args.output_prefix, name)
        df.to_csv(path, index=None)


if __name__ == '__main__':
    main()
