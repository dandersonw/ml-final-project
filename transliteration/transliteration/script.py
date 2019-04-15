import pandas as pd
import numpy as np

import abc
import itertools
import pkg_resources
import unidecode
import unicodedata

NUM_SPECIAL_TOKENS = 3
KATAKANA_BLOCK_START = 0x30A0
KATAKANA_BLOCK_END = 0x30FF  # inclusive


class Script(abc.ABC):
    @property
    @abc.abstractmethod
    def id_code():
        pass

    @property
    @abc.abstractmethod
    def join_char():
        """Character to be placed between all other characters when printing this
        script

        """
        pass

    @property
    @abc.abstractmethod
    def word_separator_char():
        """Character to be placed between words when printing this script"""
        pass

    @property
    def vocab_size(self):
        return self._vocab_size + NUM_SPECIAL_TOKENS

    @property
    @abc.abstractmethod
    def _vocab_size():
        pass

    @property
    def ins_cost(self):
        """Cost to insert or delete a character (for edit distance).

        N.B: indices are the script subclass indices

        """
        if type(self._ins_cost) is str and self._ins_cost == 'default':
            return np.ones([self._vocab_size],
                           dtype=np.float32)
        return self._ins_cost

    @property
    @abc.abstractmethod
    def _ins_cost():
        pass

    @property
    def sub_cost(self):
        """Cost to substitute a character for another (for edit distance).

        N.B: indices are the script subclass indices

        """
        if type(self._sub_cost) is str and self._sub_cost == 'default':
            return np.ones([self._vocab_size, self._vocab_size],
                           dtype=np.float32)
        return self._sub_cost

    @property
    @abc.abstractmethod
    def _sub_cost():
        pass

    def _intern_special(self, char):
        if char == '<end>':
            return self.vocab_size - 1
        elif char == '<start>':
            return self.vocab_size - 2
        elif char == 0:
            return '<pad>'
        return None

    def intern_char(self, char):
        special = self._intern_special(char)
        if special is not None:
            return special
        if not self._char_in_range(char):
            raise ValueError('{} is not a valid {} character!'
                             .format(char, self.id_code))
        result = self._intern_char(char)
        result += 1  # 0 is padding
        return result

    @abc.abstractmethod
    def _char_in_range(self, char):
        pass

    @abc.abstractmethod
    def _intern_char(self, char):
        pass

    def _deintern_special(self, interned):
        if interned == self.vocab_size - 1:
            return '<end>'
        elif interned == self.vocab_size - 2:
            return '<start>'
        elif interned == 0:
            return '<pad>'
        return None

    def deintern_char(self, interned):
        special = self._deintern_special(interned)
        if special is not None:
            return special
        interned -= 1
        return self._deintern_char(interned)

    @abc.abstractmethod
    def _deintern_char(self, char):
        pass

    @abc.abstractmethod
    def preprocess_string(self, string):
        pass


class English(Script):
    id_code = 'en'
    join_char = ''
    word_separator_char = ' '
    special_dict = {k: i + 26 for i, k in enumerate({' '})}
    reverse_special_dict = {i: k for k, i in special_dict.items()}
    _vocab_size = 26 + len(special_dict)
    _ins_cost = 'default'
    _sub_cost = 'default'

    def _char_in_range(self, char):
        return (ord(char) <= ord('z') and ord(char) >= ord('a')
                or char in self.special_dict)

    def _intern_char(self, char):
        if char in self.special_dict:
            return self.special_dict[char]
        return ord(char) - ord('a')

    def _deintern_char(self, interned):
        if interned in self.reverse_special_dict:
            return self.reverse_special_dict[interned]
        return chr(interned + ord('a'))

    def preprocess_string(self, string):
        basic = unidecode.unidecode(string)
        lower = basic.lower()
        return lower


def _katakana_ins_cost():
    r = np.ones([KATAKANA_BLOCK_END - KATAKANA_BLOCK_START + 1], np.float32)
    half_cost = {'ッ', 'ー'}
    for c in half_cost:
        r[ord(c) - KATAKANA_BLOCK_START] = 0.5
    return r


def _katakana_sub_cost():
    size = KATAKANA_BLOCK_END - KATAKANA_BLOCK_START + 1
    r = np.ones([size, size], np.float32)
    table = pd.read_csv(pkg_resources.resource_stream('transliteration.resources',
                                                      'katakana-table.csv'),
                        index_col='cons')
    for cons, row in table.iterrows():
        if cons == '∅':  # vowel mora
            continue
        row = row[pd.notnull(row)]
        for a, b in itertools.permutations(row, 2):
            a = ord(a) - KATAKANA_BLOCK_START
            b = ord(b) - KATAKANA_BLOCK_START
            r[a, b] = 0.5
    for vowel, row in table.T.iterrows():
        row = row[pd.notnull(row)]
        row = row.drop('∅')
        for a, b in itertools.permutations(row, 2):
            a = ord(a) - KATAKANA_BLOCK_START
            b = ord(b) - KATAKANA_BLOCK_START
            r[a, b] = 0.5
    return r


class Katakana(Script):
    id_code = 'ja'
    join_char = ''
    word_separator_char = ''
    _vocab_size = KATAKANA_BLOCK_END - KATAKANA_BLOCK_START + 1
    _ins_cost = _katakana_ins_cost()
    _sub_cost = _katakana_sub_cost()

    def _char_in_range(self, char):
        return (ord(char) >= KATAKANA_BLOCK_START
                and ord(char) < KATAKANA_BLOCK_END)

    def _intern_char(self, char):
        return ord(char) - KATAKANA_BLOCK_START

    def _deintern_char(self, interned):
        return chr(interned + KATAKANA_BLOCK_START)

    def preprocess_string(self, string):
        return unicodedata.normalize('NFKC', string)


class CMUPronunciation(Script):
    id_code = 'cmu'
    join_char = ' '
    word_separator_char = ''
    intern_dict = {k: i for i, k in
                   enumerate(str(pkg_resources.resource_string('transliteration.resources',
                                                               'cmudict-0.7b.symbols'),
                                 encoding='utf8')
                             .split('\n'))}
    reverse_intern_dict = {v: k for k, v in intern_dict.items()}
    _vocab_size = len(intern_dict)
    _ins_cost = 'default'
    _sub_cost = 'default'

    def _char_in_range(self, char):
        return char in self.intern_dict

    def _intern_char(self, char):
        return self.intern_dict[char]

    def _deintern_char(self, char):
        return self.reverse_intern_dict[char]

    def preprocess_string(self, string):
        return string.split(' ')


SCRIPTS = dict()
for script_class in [English, Katakana, CMUPronunciation]:
    script = script_class()
    SCRIPTS[script.id_code] = script
