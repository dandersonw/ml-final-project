import tensorflow as tf

import unidecode
import unicodedata


NUM_SPECIAL_TOKENS = 3
KATAKANA_BLOCK_START = 0x30A0
KATAKANA_BLOCK_END = 0x30FF  # inclusive
KATAKANA_VOCAB_SIZE = KATAKANA_BLOCK_END - KATAKANA_BLOCK_START + 1 + NUM_SPECIAL_TOKENS
ENGLISH_VOCAB_SIZE = 26 + NUM_SPECIAL_TOKENS


def intern_special(vocab_size, char):
    if char == '<end>':
        return vocab_size - 1
    if char == '<start>':
        return vocab_size - 2
    return None


def deintern_special(vocab_size, interned):
    if interned == vocab_size - 1:
        return '<end>'
    if interned == vocab_size - 2:
        return '<start>'
    return None


def intern_katakana_char(char):
    special = intern_special(KATAKANA_VOCAB_SIZE, char)
    if special is not None:
        return special
    if ord(char) < KATAKANA_BLOCK_START or ord(char) > KATAKANA_BLOCK_END:
        raise ValueError('"{}" is not a katakana character!'.format(char))
    return ord(char) - KATAKANA_BLOCK_START + 1  # reserve 0 for padding


def deintern_katakana_char(interned):
    special = deintern_special(KATAKANA_VOCAB_SIZE, interned)
    if special is not None:
        return special
    idx = interned - 1  # 0 was reserved for padding
    return chr(KATAKANA_BLOCK_START + idx)


def preprocess_ja_string(ja):
    return unicodedata.normalize('NFKC', ja)


def intern_en_char(char):
    special = intern_special(ENGLISH_VOCAB_SIZE, char)
    if special is not None:
        return special
    if ord(char) > ord('z') or ord(char) < ord('a'):
        return ValueError('"{}" is not a valid English character!'.format(char))
    return ord(char) - ord('a') + 1


def deintern_en_char(interned):
    special = deintern_special(ENGLISH_VOCAB_SIZE, interned)
    if special is not None:
        return special
    if interned == ENGLISH_VOCAB_SIZE - 1:
        return '<end>'
    return chr(interned - 1 + ord('a'))


def preprocess_english_string(en):
    basic = unidecode.unidecode(en)
    lower = basic.lower()
    return lower


def create_datum(*, en, ja):
    en = [intern_en_char(c) for c in preprocess_english_string(en)]
    ja = [intern_katakana_char(c) for c in (list(preprocess_ja_string(ja)) + ['<end>'])] 
    length_en = len(en)
    length_ja = len(ja)
    return {'en': en,
            'ja': ja,
            'length_en': length_en,
            'length_ja': length_ja}


def datum_to_tf_example(datum: dict) -> tf.train.SequenceExample:
    example = tf.train.SequenceExample()
    example.context.feature['length_en'].int64_list.value.append(datum['length_en'])
    example.context.feature['length_ja'].int64_list.value.append(datum['length_ja'])
    tokens = example.feature_lists.feature_list['en'].feature
    for t in datum['en']:
        tokens.add().int64_list.value.append(t)
    tokens = example.feature_lists.feature_list['ja'].feature
    for t in datum['ja']:
        tokens.add().int64_list.value.append(t)
    return example


def parse_tf_example(example):
    context_features = {'length_en': tf.FixedLenFeature([], dtype=tf.int64),
                        'length_ja': tf.FixedLenFeature([], dtype=tf.int64)}
    sequence_features = {'en': tf.FixedLenSequenceFeature([], dtype=tf.int64),
                         'ja': tf.FixedLenSequenceFeature([], dtype=tf.int64)}

    context_parsed, sequence_parsed \
        = tf.parse_single_sequence_example(context_features=context_features,
                                           sequence_features=sequence_features,
                                           serialized=example)

    return {**{k: context_parsed[k] for k in context_features},
            **{k: sequence_parsed[k] for k in sequence_features}}


def make_dataset(path, batch_size=128) -> tf.data.Dataset:
    with tf.name_scope(path):
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(lambda d: parse_tf_example(d))
        dataset = dataset.shuffle(buffer_size=1000)
        padding_shapes = {'length_en': [],
                          'en': [None],
                          'length_ja': [],
                          'ja': [None]}
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes=padding_shapes)
        dataset = dataset.prefetch(10)
        return dataset
