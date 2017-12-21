import numpy as np
import pandas as pd

from sklearn_models import MultNB, BernNB, SVM
from keras_models.fchollet_cnn import FCholletCNN
from keras_models.mlp import MLP
from benchmarks import benchmark
# import globalVar as gl
import ttm.globalVar as gl
gl._init()

# datasets = [
#     "../data/myOwn/20_newsgroups",
#     "../data/myOwn/20news-18828",
#     "../data/myOwn/20news-bydate"
# ]

datasets = [
    '../data/20ng-all-terms.txt',
    '../data/20ng-no-short.txt',
    '../data/20ng-no-stop.txt',
    '../data/20ng-stemmed.txt',
    '../data/r52-all-terms.txt',
    '../data/r52-no-short.txt',
    '../data/r52-no-stop.txt',
    '../data/r52-stemmed.txt',
    '../data/r8-all-terms.txt',
    '../data/r8-no-short.txt',
    '../data/r8-no-stop.txt',
    '../data/r8-stemmed.txt',
    '../data/webkb-stemmed.txt'
]

models = [
    # (FCholletCNN, {'dropout_rate': 0.5, 'embedding_dim': 37, 'units': 400, 'epochs': 30, "POS_tagging" : True, "Weight": 0.5}, "CNN 37D"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 0}, "CNN GloVe"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 0.1}, "CNN GloVe"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 0.2}, "CNN GloVe"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 0.3}, "CNN GloVe"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 0.4}, "CNN GloVe"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 0.5}, "CNN GloVe"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 0.6}, "CNN GloVe"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 0.7}, "CNN GloVe"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 0.8}, "CNN GloVe"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 0.9}, "CNN GloVe"),
    (FCholletCNN, {'dropout_rate': 0.5, 'epochs': 20, 'units': 400,
                   'embeddings_path': '../data/glove.6B/glove.6B.100d.txt', "pos_tagging": True, "weight": 1}, "CNN GloVe")
    # (MLP, {'layers': 1, 'units': 360, 'dropout_rate': 0.87, 'epochs': 12, 'max_vocab_size': 22000, "POS_tagging" : True}, "MLP 1x360"),
    # (MLP, {'layers': 2, 'units': 180, 'dropout_rate': 0.6, 'epochs': 5, 'max_vocab_size': 22000, "POS_tagging" : True}, "MLP 2x180"),
    # (MLP, {'layers': 3, 'dropout_rate': 0.2, 'epochs': 20, "POS_tagging" : True}, "MLP 3x512"),
    # (MultNB, {'tfidf': True, "POS_tagging" : True}, "MNB tfidf"),
    # (MultNB, {'tfidf': True, 'ngram_n': 2, "POS_tagging" : True}, "MNB tfidf 2-gr"),
    # (MultNB, {'tfidf': True, 'ngram_n': 3, "POS_tagging" : True}, "MNB tfidf 3-gr"),
    # (BernNB, {'tfidf': True, "POS_tagging" : True}, "BNB tfidf"),
    # (MultNB, {'tfidf': False, "POS_tagging" : True}, "MNB"),
    # (MultNB, {'tfidf': False, 'ngram_n': 2, "POS_tagging" : True}, "MNB 2-gr"),
    # (BernNB, {'tfidf': False, "POS_tagging" : True}, "BNB"),
    # (SVM, {'tfidf': True, 'kernel': 'linear', "POS_tagging" : True}, "SVM tfidf"),
    # (SVM, {'tfidf': True, 'kernel': 'linear', 'ngram_n': 2, "POS_tagging" : True}, "SVM tfidf 2-gr"),
    # (SVM, {'tfidf': False, 'kernel': 'linear', "POS_tagging" : True}, "SVM"),
    # (SVM, {'tfidf': False, 'kernel': 'linear', 'ngram_n': 2, "POS_tagging" : True}, "SVM 2-gr")
]

results_path = "document_pos_tag_nn_vt_results.csv"

if __name__ == '__main__':
    records = []
    for data_path in datasets:
        print
        print data_path

        for model_class, params, model_name in models:
            scores, times = benchmark(model_class, data_path, params, 5)
            model_str = str(model_class(**params))
            print '%.3f' % np.mean(scores), model_str
            for score, time in zip(scores, times):
                records.append({
                    'model': model_str,
                    'dataset_name': data_path.split("/")[-1],
                    'accuracy': score,
                    'time': time,
                    'model_name': model_name
                })

    pd.DataFrame(records).to_csv(results_path, index=False, mode="a")
