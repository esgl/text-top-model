import numpy as np
from time import time

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from prepare_data import prepare_dataset, cache
import ttm.globalVar as gl

@cache
def benchmark(model_class, data_path, model_params=None, iters=1):
    """benchmarks a given model on a given dataset
    Instantiates the model with given parameters.
    :param model_class: class of the model to instantiate
    :param data_path: path to file with dataset
    :param model_params: optional dictionary with model parameters
    :param iters: how many times to benchmark
    :param return_time: if true, returns list of running times in addition to scores
    :return: tuple (accuracy scores, running times)
    """
    if model_params is None:
        model_params = {}

    model_params["pos_tagging"] = model_params["pos_tagging"] if model_params.has_key("pos_tagging") else None
    gl.set_value("POS_tagging", model_params["pos_tagging"])

    model_params["weight"] = model_params["weight"] if model_params.has_key("weight") else 0
    assert model_params["weight"] >= 0 and model_params["weight"] <=1, "weight must bewteen 0 and 1"
    gl.set_value("weight", model_params["weight"])

    # X, y, vocab, label_encoder = prepare_dataset(data_path)
    X, y, vocab, label_encoder = prepare_dataset(data_path)
    class_count = len(label_encoder.classes_)
    model_params['vocab_size'] = len(vocab)
    model_params['vocab'] = vocab
    model_params['class_count'] = class_count

    scores = []
    times = []
    for i in range(iters):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)

        model = model_class(**model_params)
        start = time()
        preds = model.fit(X_train, y_train).predict(X_test)
        end = time()
        scores.append(accuracy_score(preds, y_test))
        times.append(end - start)
    return scores, times


@cache
def benchmark_with_early_stopping(model_class, data_path, model_params=None):
    """same as benchmark but fits with validation data to allow the model to do early stopping
    Works with all models from keras_models
    :param model_class: class of the model to instantiate, must have fit(X, y, validation_data)
        method and 'history' attribute
    :param data_path: path to file with dataset
    :param model_params: optional dictionary with model parameters
    :return: best_loss, best_score, best_epoch
    """
    if model_params is None:
        model_params = {}

    X, y, vocab, label_encoder = prepare_dataset(data_path)
    class_count = len(label_encoder.classes_)
    model_params['vocab_size'] = len(vocab)
    model_params['vocab'] = vocab
    model_params['class_count'] = class_count
    model = model_class(**model_params)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    model.fit(X_train, y_train, validation_data=[X_test, y_test])
    best_loss = np.min(model.history.history['val_loss'])
    best_acc = np.max(model.history.history['val_acc'])
    best_epoch = np.argmin(model.history.history['val_loss']) + 1

    print model, "acc", best_acc, "loss",  best_loss, "epochs", best_epoch
    return best_loss, best_acc, best_epoch
