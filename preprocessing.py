from typing import Sequence
from pandas import DataFrame
import numpy as np
from collections import Counter

UNK_TOKEN = "<unk>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"


class CharPreprocessor(object):

    def __init__(self, text_col: str, max_vocab_size: int = None, start_end_token=True, unk_token=True):

        self.text_col = text_col
        self.max_features = max_vocab_size
        self.start_end_token = start_end_token
        self.unk_token = unk_token

        self.vocabulary = None
        self.inv_vocabulary = None
        self.is_fitted = False

    @property
    def shape(self) -> Sequence[int]:
        return [None, len(self.vocabulary)]

    def fit(self, data: DataFrame) -> DataFrame:
        """
        :param data: A pandas DataFrame of training data to fit
        :return:
        """
        counter = Counter()
        for text in data[self.text_col]:
            counter.update(text)

        chars = counter.most_common(self.max_features)
        if self.unk_token:
            chars.append((UNK_TOKEN, None))
        if self.start_end_token:
            chars.append((START_TOKEN, None))
            chars.append((END_TOKEN, None))

        self.vocabulary = {char: i for i, (char, _) in enumerate(chars)}
        self.inv_vocabulary = {v: k for k, v in self.vocabulary.items()}
        self.is_fitted = True

    def transform(self, data):
        decoder_seq, decoder_lengths = self.transform_texts(data[self.text_col].astype(str), adjust_left=True, reverse=False, start_token=False, end_token=True)
        decoder_input = decoder_seq[:,:-1] # Don't feed END token
        decoder_targets = decoder_seq[:,1:] # Don't predict START token
        return (decoder_input, decoder_lengths - 1), decoder_targets

    def transform_texts(self, texts, adjust_left=False, reverse=False, start_token=None, end_token=None, padding_int=0):
        if start_token is None or not self.start_end_token: start_token = self.start_end_token
        if end_token is None or not self.start_end_token: end_token = self.start_end_token
        if not self.unk_token:
            # Filter chars not in vocab
            texts = ["".join(filter(lambda x: x in self.vocabulary, text)) for text in texts]

        lens = np.array(list(map(lambda x: len(x), texts)))
        X = np.full((len(texts), max(lens) + start_token + end_token), padding_int, dtype='int32')

        if start_token:
            start_token_idxs = 0 if adjust_left else max(lens) - lens
            X[np.arange(X.shape[0]), start_token_idxs] = self.vocabulary[START_TOKEN]
        if end_token:
            end_token_idxs = lens + start_token if adjust_left else X.shape[1] - 1
            X[np.arange(X.shape[0]), end_token_idxs] = self.vocabulary[END_TOKEN]

        def opt_reverse(iter):
            if reverse:
                return reversed(iter)
            else:
                return iter

        for i, text in enumerate(texts):
            adj = start_token if adjust_left else max(lens) - lens[i] + start_token
            for j, char in enumerate(opt_reverse(text)):
                if char in self.vocabulary:
                    X[i, j+adj] = self.vocabulary[char]
                elif self.unk_token:
                    X[i, j+adj] = self.vocabulary[UNK_TOKEN]
        return X, lens + start_token + end_token


def test_char_preprocessor():

    class CharPreprocessorTest(CharPreprocessor):
        def transform(self, data: DataFrame):
            pass

    data = DataFrame([["aabc"], ["ab"]], columns=["text"])

    # No start/end/unk token
    p = CharPreprocessorTest(text_col="text", max_vocab_size=2, start_end_token=False, unk_token=False)
    p.fit(data)

    a = p.vocabulary["a"]
    b = p.vocabulary["b"]
    O = 0

    assert p.shape[1] == 2, "Shape doesn't match"
    assert np.array_equal(p.transform_texts(data.text, adjust_left=True, reverse=False, start_token=True)[0],
                            np.array([[a, a, b],
                                      [a, b, O]]))
    assert np.array_equal(p.transform_texts(data.text, adjust_left=True, reverse=True, end_token=True)[0],
                            np.array([[b, a, a],
                                      [b, a, O]]))
    assert np.array_equal(p.transform_texts(data.text, adjust_left=False, reverse=True)[0],
                          np.array([[b, a, a],
                                    [O, b, a]]))

    # With start/end/unk/padding tokens
    p = CharPreprocessorTest(text_col="text", max_vocab_size=2, start_end_token=True, unk_token=True)
    p.fit(data)

    a = p.vocabulary["a"]
    b = p.vocabulary["b"]
    i = p.vocabulary[START_TOKEN]
    j = p.vocabulary[END_TOKEN]
    u = p.vocabulary[UNK_TOKEN]
    O = -1

    assert p.shape[1] == 5, "Shape doesn't match"
    assert np.array_equal(p.transform_texts(data.text, adjust_left=True, reverse=False, start_token=False, end_token=False, padding_int=O)[0],
                            np.array([[a, a, b, u],
                                      [a, b, O, O]]))
    assert np.array_equal(p.transform_texts(data.text, adjust_left=True, reverse=False, padding_int=O, end_token=False)[0],
                            np.array([[i, a, a, b, u],
                                      [i, a, b, O, O]]))
    assert np.array_equal(p.transform_texts(data.text, adjust_left=True, reverse=True, padding_int=O)[0],
                            np.array([[i, u, b, a, a, j],
                                      [i, b, a, j, O, O]]))
    assert np.array_equal(p.transform_texts(data.text, adjust_left=False, reverse=False, padding_int=O)[0],
                            np.array([[i, a, a, b, u, j],
                                      [O, O, i, a, b, j]]))


if __name__ == "__main__":
    test_char_preprocessor()