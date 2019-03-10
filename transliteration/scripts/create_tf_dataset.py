import tensorflow as tf
import pandas as pd

import argparse

from transliteration import data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()

    input_data = pd.read_csv(args.input_file, keep_default_na=False)

    with tf.python_io.TFRecordWriter(args.output_file) as writer:
        for _, row in input_data.iterrows():
            datum = data.create_datum(**row)
            example = data.datum_to_tf_example(datum)
            writer.write(example.SerializeToString())


if __name__ == '__main__':
    main()
