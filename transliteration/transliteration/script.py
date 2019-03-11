import unidecode
import unicodedata

import abc

NUM_SPECIAL_TOKENS = 3
KATAKANA_BLOCK_START = 0x30A0
KATAKANA_BLOCK_END = 0x30FF  # inclusive


class Script(abc.ABC):
    @property
    @abc.abstractmethod
    def id_code():
        pass

    @property
    def vocab_size(self):
        return self._vocab_size + NUM_SPECIAL_TOKENS

    @property
    @abc.abstractmethod
    def _vocab_size():
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
    _vocab_size = 26

    def _char_in_range(self, char):
        return ord(char) <= ord('z') and ord(char) >= ord('a')

    def _intern_char(self, char):
        return ord(char) - ord('a')

    def _deintern_char(self, interned):
        return chr(interned + ord('a'))

    def preprocess_string(self, string):
        basic = unidecode.unidecode(string)
        lower = basic.lower()
        return lower


class Katakana(Script):
    id_code = 'ja'
    _vocab_size = KATAKANA_BLOCK_END - KATAKANA_BLOCK_START + 1

    def _char_in_range(self, char):
        return (ord(char) >= KATAKANA_BLOCK_START
                and ord(char) < KATAKANA_BLOCK_END)

    def _intern_char(self, char):
        return ord(char) - KATAKANA_BLOCK_START

    def _deintern_char(self, interned):
        return chr(interned + KATAKANA_BLOCK_START)

    def preprocess_string(self, string):
        return unicodedata.normalize('NFKC', string)


SCRIPTS = dict()
for script_class in [English, Katakana]:
    script = script_class()
    SCRIPTS[script.id_code] = script
