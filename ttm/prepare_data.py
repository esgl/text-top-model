import numpy as np
import os
import sys

from sklearn.preprocessing import LabelEncoder
from collections import Counter
from joblib import Memory

from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

import ttm.globalVar as gl
cache = Memory('cache').cache


def read_dataset(path):
    X, y = [], []
    words_nn_vb_set = set()
    with open(path, "rb") as infile:
        for line in infile:
            label, text = line.split("\t")
            text = text.strip()
            if len(text) == 0:
                continue
            # texts are already tokenized, just split on space
            # in a real case we would use e.g. spaCy for tokenization
            # and maybe remove stopwords etc.
            if gl.get_value("pos_tagging"):

                text_pos = pos_tag(word_tokenize(text))
                pos_nn_vt_tags = set(["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])
                for i in xrange(len(text_pos)):
                    word_pos_tag = text_pos[i][1]
                    if word_pos_tag in pos_nn_vt_tags:
                        word = text_pos[i][0]
                        words_nn_vb_set.add(word)

            X.append(text.split())
            y.append(label)
    if gl.get_value("pos_tagging"):
        gl.set_value("Words_nn_vb_set", words_nn_vb_set)
    X, y = np.array(X), np.array(y)
    print "total examples %s" % len(y)
    return X, y


@cache
def prepare_dataset(path):
    X, y = read_dataset(path)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(y)
    word_counts = Counter(w for text in X for w in text)
    vocab = [''] + [w for (w, _) in sorted(word_counts.items(), key=lambda (_, c): -c)]
    word2ind = {w: i for i, w in enumerate(vocab)}
    X = np.array([[word2ind[w] for w in tokens] for tokens in X])
    return X, labels, vocab, label_encoder
