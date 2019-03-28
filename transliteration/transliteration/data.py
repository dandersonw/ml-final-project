import tensorflow as tf

import typing

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


def parse_tf_example(example, script_names: typing.List[str]):
    context_features = {'length_{}'.format(s):
                        tf.FixedLenFeature([], dtype=tf.int64)
                        for s in script_names}
    sequence_features = {s: tf.FixedLenSequenceFeature([], dtype=tf.int64)
                         for s in script_names}

    context_parsed, sequence_parsed \
        = tf.parse_single_sequence_example(context_features=context_features,
                                           sequence_features=sequence_features,
                                           serialized=example)

    result = {**{k: context_parsed[k] for k in context_features},
              **{k: sequence_parsed[k] for k in sequence_features}}
    for s in script_names:
        result['length_{}'.format(s)] = tf.cast(result['length_{}'.format(s)], tf.int32)
    return result


def _append_end_token(example, scripts):
    result = {**example}
    for scriptt in scripts:
        end_token = SCRIPTS[scriptt].intern_char('<end>')
        result[scriptt] = tf.concat([example[scriptt],
                                     tf.constant(end_token,
                                                 shape=[1],
                                                 dtype=tf.int64)],
                                    axis=0)
        len_key = 'length_{}'.format(scriptt)
        result[len_key] = example[len_key] + 1
    return result


def _combine_words_augmentation(dataset: tf.data.Dataset,
                                script_names: typing.List[str],
                                padding_shapes,
                                proportion):
    def make_idxs(join_token_exists, max_i, max_len):
        def result(tensors):
            length, i = tensors
            idxs = tf.stack([tf.tile(tf.expand_dims(i, 0),
                                     tf.expand_dims(length, 0)),
                             tf.range(length)],
                            axis=1)
            if join_token_exists:
                fake_seq_idx = tf.expand_dims(max_i, 0)
                sep_token_idx = tf.stack([fake_seq_idx,
                                          tf.zeros_like(fake_seq_idx)],
                                         axis=1)
                idxs = tf.cond(i < max_i - 1,
                               lambda: tf.concat([idxs,
                                                  sep_token_idx],
                                                 axis=0),
                               lambda: idxs)
                length = tf.cond(i < max_i - 1,
                                 lambda: length + 1,
                                 lambda: length)
            idxs = tf.pad(idxs,
                          [[0, max_len - length], [0, 0]],
                          constant_values=-1)
            return (idxs, length)
        return result

    def concat(batch):
        result = dict()
        for scriptt in script_names:
            sep_char = SCRIPTS[scriptt].word_separator_char
            sep_token = SCRIPTS[scriptt].intern_char(sep_char) if sep_char else None
            tokens = batch[scriptt]
            length_key = 'length_{}'.format(scriptt)
            lengths = batch[length_key]

            if sep_token is not None:
                fake_seq_shape = tf.expand_dims(tf.shape(tokens)[1], 0)
                sep_token = tf.constant(sep_token, shape=[1], dtype=tf.int64)
                fake_seq = tf.tile(sep_token, fake_seq_shape)
                tokens = tf.concat([tokens,
                                    tf.expand_dims(fake_seq, 0)],
                                   axis=0)

            idxs, new_lengths = tf.map_fn(make_idxs(sep_token is not None,
                                                    tf.shape(lengths)[0],
                                                    tf.reduce_max(lengths) + 1),
                                          (lengths,
                                           tf.range(tf.shape(lengths)[0])),
                                          infer_shape=False,
                                          dtype=(tf.int32, tf.int32))
            idxs = tf.reshape(idxs, [-1, 2])
            wasnt_padding = tf.reduce_any(tf.greater(idxs, -1), axis=1)
            idxs = tf.boolean_mask(idxs, wasnt_padding)
            result[scriptt] = tf.gather_nd(tokens, idxs)
            result[length_key] = tf.reduce_sum(new_lengths)
        return result

    def maybe_concat(batch):
        cond = tf.less(tf.random.uniform([], dtype=tf.float32),
                       tf.constant(proportion, shape=[]))
        element = tf.cond(cond,
                          true_fn=lambda: concat(batch),
                          false_fn=lambda: {k: v[0] for k, v in batch.items()})
        return element
    return dataset.padded_batch(2, padded_shapes=padding_shapes)\
                  .map(maybe_concat)


def make_dataset(path,
                 from_script: str,
                 to_script: str,
                 combine_words_proportion=None,
                 batch_size=128) -> tf.data.Dataset:
    script_names = [from_script, to_script]
    padding_shapes = {'length_{}'.format(from_script): [],
                      '{}'.format(from_script): [None],
                      'length_{}'.format(to_script): [],
                      '{}'.format(to_script): [None]}
    with tf.name_scope(path):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(lambda d: parse_tf_example(d, script_names))
        dataset = dataset.shuffle(buffer_size=100000)
        if combine_words_proportion is not None:
            dataset = _combine_words_augmentation(dataset,
                                                  script_names,
                                                  padding_shapes,
                                                  combine_words_proportion)
        dataset = dataset.map(lambda d: _append_end_token(d, script_names))
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=padding_shapes)
        return dataset
