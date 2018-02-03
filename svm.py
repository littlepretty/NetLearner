import numpy as np
from sklearn import svm
from netlearner.utils import standard_scale, measure_prediction
from preprocess import unsw, nslkdd
import sys


def load_nsl_dataset():
    nslkdd.generate_dataset(False, True, model_dir)
    raw_train_dataset = np.load(data_dir + 'train_dataset.npy')
    train_labels = np.load(data_dir + 'train_labels.npy')
    raw_valid_dataset = np.load(data_dir + 'valid_dataset.npy')
    raw_test_dataset = np.load(data_dir + 'test_dataset.npy')
    test_labels = np.load(data_dir + 'test_labels.npy')
    # train_dataset, valid_dataset, test_dataset = min_max_normalize(
    #    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
    # print('Min-Max normalizing dataset')
    train_dataset, valid_dataset, test_dataset = standard_scale(
        raw_train_dataset, raw_valid_dataset, raw_test_dataset)
    print('Mean normalizing dataset')
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    return (train_dataset, train_labels, test_dataset, test_labels)


def load_unsw_dataset():
    unsw.generate_dataset(True, model_dir)
    raw_train_dataset = np.load(data_dir + 'train_dataset.npy')
    train_labels = np.load(data_dir + 'train_labels.npy')
    raw_valid_dataset = np.load(data_dir + 'valid_dataset.npy')
    raw_test_dataset = np.load(data_dir + 'test_dataset.npy')
    test_labels = np.load(data_dir + 'test_labels.npy')
    # train_dataset, valid_dataset, test_dataset = min_max_normalize(
    #    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
    # print('Min-Max normalizing dataset')
    train_dataset, valid_dataset, test_dataset = standard_scale(
        raw_train_dataset, raw_valid_dataset, raw_test_dataset)
    print('Mean normalizing dataset')
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    return (train_dataset, train_labels, test_dataset, test_labels)


def evaluate(predictions, labels, description):
    predicted = np.zeros_like(labels)
    for (i, x) in enumerate(predictions):
        predicted[i][x] = 1.0

    measure_prediction(predicted, labels, data_dir, description)


def LinearSVM(train_dataset, train_labels, test_dataset, test_labels):
    print('Use linear SVM')
    y = np.argmax(train_labels, 1)
    clf = svm.LinearSVC(class_weight=class_weight, verbose=True)
    clf.fit(train_dataset, y)

    train_prediction = clf.predict(train_dataset)
    evaluate(train_prediction, train_labels, 'LinearSVM Train')
    test_prediction = clf.predict(test_dataset)
    evaluate(test_prediction, test_labels, 'LinearSVM Test')


def NonLinearSVM(train_dataset, train_labels, test_dataset, test_labels):
    print('Use non-linear SVM')
    y = np.argmax(train_labels, 1)
    clf = svm.SVC(class_weight=class_weight, verbose=True)
    clf.fit(train_dataset, y)

    train_prediction = clf.predict(train_dataset)
    evaluate(train_prediction, train_labels, 'NonLinearSVM Train')
    test_prediction = clf.predict(test_dataset)
    evaluate(test_prediction, test_labels, 'NonLinearSVM Test')


if __name__ == '__main__':
    np.random.seed(5)
    model_dir = 'SVM/'
    if sys.argv[1] == 'unsw':
        class_weight = {0: 1.0, 1: 1.0}
        data_dir = model_dir + 'UNSW/'
        train, train_labels, test, test_labels = load_unsw_dataset()
    else:
        class_weight = {0: 1.0, 1: 4.0, 2: 1.0, 3: 16.0, 4: 4.0}
        data_dir = model_dir + 'NSLKDD/'
        train, train_labels, test, test_labels = load_nsl_dataset()

    NonLinearSVM(train, train_labels, test, test_labels)
    LinearSVM(train, train_labels, test, test_labels)
