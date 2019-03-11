import tensorflow as tf

from .script import SCRIPTS


def create_datum(**kwargs):
    result = dict()
    for script_name, script_seq in kwargs.items():
        scriptt = SCRIPTS[script_name]
        script_seq = [scriptt.intern_char(c)
                      for c in scriptt.preprocess_string(script_seq)]
        result[script_name] = script_seq
        result['length_{}'.format(script_name)] = len(script_seq)
    return result


def datum_to_tf_example(datum: dict) -> tf.train.SequenceExample:
    example = tf.train.SequenceExample()
    for key, value in datum.items():
        if key.startswith('length_'):
            example.context.feature[key].int64_list.value.append(value)
        elif key in SCRIPTS:
            tokens = example.feature_lists.feature_list[key].feature
            for t in value:
                tokens.add().int64_list.value.append(t)
    return example


def parse_tf_example(example, script_names):
    context_features = {'length_{}'.format(s):
                        tf.FixedLenFeature([], dtype=tf.int64)
                        for s in script_names}
    sequence_features = {s: tf.FixedLenSequenceFeature([], dtype=tf.int64)
                         for s in script_names}

    context_parsed, sequence_parsed \
        = tf.parse_single_sequence_example(context_features=context_features,
                                           sequence_features=sequence_features,
                                           serialized=example)

    return {**{k: context_parsed[k] for k in context_features},
            **{k: sequence_parsed[k] for k in sequence_features}}


def _append_end_token(example, to_script):
    result = {**example}
    end_token = SCRIPTS[to_script].intern_char('<end>')
    result[to_script] = tf.concat([example[to_script],
                                   tf.constant(end_token,
                                               shape=[1],
                                               dtype=tf.int64)],
                                  axis=0)
    len_key = 'length_{}'.format(to_script)
    result[len_key] = example[len_key] + 1
    return result


def make_dataset(path,
                 from_script,
                 to_script,
                 batch_size=128) -> tf.data.Dataset:
    with tf.name_scope(path):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(lambda d: parse_tf_example(d, [from_script, to_script]))
        dataset = dataset.map(lambda d: _append_end_token(d, to_script))
        dataset = dataset.shuffle(buffer_size=1000)
        padding_shapes = {'length_{}'.format(from_script): [],
                          '{}'.format(from_script): [None],
                          'length_{}'.format(to_script): [],
                          '{}'.format(to_script): [None]}
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=padding_shapes)
        dataset = dataset.prefetch(10)
        return dataset
