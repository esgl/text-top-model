import numpy as np
from joblib import Memory

import ttm.globalVar as gl

cache = Memory('cache').cache


@cache
def get_embedding_dim(embedding_path):
    with open(embedding_path, 'rb') as f:
        return len(f.readline().split()) - 1


# @cache
def get_embedding_matrix(vocab, embedding_path):
    word2ind = {w: i for i, w in enumerate(vocab)}
    embedding_dim = get_embedding_dim(embedding_path)
    embeddings = np.random.normal(size=(len(vocab), embedding_dim))

    with open(embedding_path, 'rb') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            if word in word2ind:
                i = word2ind[word]
                vec = np.array([float(x) for x in parts[1:]])
                if gl.get_value("pos_tagging"):
                    if word in gl.get_value("Words_nn_vb_set"):
                        vec *= (1 + gl.get_value("weight"))
                    else:
                        vec *= (1 - gl.get_value("weight"))
                embeddings[i] = vec
    return embeddings
