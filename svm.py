import numpy as np
from sklearn import svm
from netlearner.utils import standard_scale
from tabulate import tabulate

np.random.seed(5)


def load_nsl_dataset():
    raw_train_dataset = np.load('NSLKDD/train_dataset_bin.npy')
    train_labels = np.load('NSLKDD/train_ref_bin.npy')
    raw_valid_dataset = np.load('NSLKDD/valid_dataset_bin.npy')
    valid_labels = np.load('NSLKDD/valid_ref_bin.npy')
    raw_test_dataset = np.load('NSLKDD/test_dataset_bin.npy')
    test_labels = np.load('NSLKDD/test_ref_bin.npy')
    # train_dataset, valid_dataset, test_dataset = min_max_normalize(
    #    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
    # print('Min-Max normalizing dataset')
    train_dataset, valid_dataset, test_dataset = standard_scale(
        raw_train_dataset, raw_valid_dataset, raw_test_dataset)
    print('Mean normalizing dataset')
    train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
    train_labels = np.concatenate((train_labels, valid_labels), axis=0)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    return (train_dataset, train_labels,
            valid_dataset, valid_labels,
            test_dataset, test_labels)


def load_unsw_dataset():
    raw_train_dataset = np.load('UNSW/train_dataset.npy')
    train_labels = np.load('UNSW/train_labels_bin.npy')
    raw_valid_dataset = np.load('UNSW/valid_dataset.npy')
    valid_labels = np.load('UNSW/valid_labels_bin.npy')
    raw_test_dataset = np.load('UNSW/test_dataset.npy')
    test_labels = np.load('UNSW/test_labels_bin.npy')
    # train_dataset, valid_dataset, test_dataset = min_max_normalize(
    #    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
    # print('Min-Max normalizing dataset')
    train_dataset, valid_dataset, test_dataset = standard_scale(
        raw_train_dataset, raw_valid_dataset, raw_test_dataset)
    print('Mean normalizing dataset')
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    return (train_dataset, train_labels,
            valid_dataset, valid_labels,
            test_dataset, test_labels)


def accuracy(y, target):
    return float(np.sum(y == target)) / float(target.shape[0])


def compute_classification_table(predictions, labels, dirname, dataset):
    num_classes = labels.shape[1]
    class_table = np.zeros((num_classes, num_classes))
    actual_class = np.argmax(labels, 1)
    for (a, p) in zip(actual_class, predictions):
        class_table[a][p] += 1

    headers = [str(i) for i in range(labels.shape[1])]
    print(tabulate(class_table, headers))

    log = open(dirname + '/performance.log', 'a')
    log.write('*****************    %s    *******************\n' % dataset)
    log.write(tabulate(class_table, headers) + '\n')
    log.write('************************************************\n')
    log.close

    return class_table


def LinearSVM(train_dataset, train_labels,
              test_dataset, test_labels):
    print('Use linear SVM')
    y = np.argmax(train_labels, 1)
    # class_weight = {0: 1, 1: 1, 2: 1, 3: 1000, 4: 1000}
    binary_class_weight = {0: 1, 1: 1}
    clf = svm.LinearSVC(class_weight=binary_class_weight, verbose=False)
    clf.fit(train_dataset, y)

    train_prediction = clf.predict(train_dataset)
    print('Trainset accuracy: %f' % accuracy(train_prediction, y))
    compute_classification_table(train_prediction, train_labels, 'LinearSVM', 'TRAIN')

    test_prediction = clf.predict(test_dataset)
    test_y = np.argmax(test_labels, 1)
    print('Testset accuracy: %f' % accuracy(test_prediction, test_y))
    compute_classification_table(test_prediction, test_labels, 'LinearSVM', 'TEST')


def NonLinearSVM(train_dataset, train_labels,
                 test_dataset, test_labels):
    print('Use non-linear SVM')
    y = np.argmax(train_labels, 1)
    # class_weight = {0: 1, 1: 1, 2: 1, 3: 1000, 4: 1000}
    binary_class_weight = {0: 1, 1: 1}
    clf = svm.SVC(class_weight=binary_class_weight, verbose=False)
    clf.fit(train_dataset, y)

    train_prediction = clf.predict(train_dataset)
    print('Trainset accuracy: %f' % accuracy(train_prediction, y))
    compute_classification_table(train_prediction, train_labels, 'NonLinearSVM', 'TRAIN')

    test_prediction = clf.predict(test_dataset)
    test_y = np.argmax(test_labels, 1)
    print('Testset accuracy: %f' % accuracy(test_prediction, test_y))
    compute_classification_table(test_prediction, test_labels, 'NonLinearSVM', 'TEST')


# train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load_nsl_dataset()
train_dataset, train_labels, _, _, test_dataset, test_labels = load_unsw_dataset()
NonLinearSVM(train_dataset, train_labels, test_dataset, test_labels)
LinearSVM(train_dataset, train_labels, test_dataset, test_labels)
