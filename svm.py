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
    train_labels = np.load('UNSW/train_labels.npy')
    raw_valid_dataset = np.load('UNSW/valid_dataset.npy')
    valid_labels = np.load('UNSW/valid_labels.npy')
    raw_test_dataset = np.load('UNSW/test_dataset.npy')
    test_labels = np.load('UNSW/test_labels.npy')
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


def compute_classification_table(predictions, labels, dirname, dataset,
                                 hyperparameter=None, print_hyper=False):
    num_classes = labels.shape[1]
    class_table = np.zeros((num_classes, num_classes))
    actual_class = np.argmax(labels, 1)
    for (a, p) in zip(actual_class, predictions):
        class_table[a][p] += 1

    e = 1e-26
    recall = [class_table[i][i] / (np.sum(class_table[i, :]) + e)
              for i in range(num_classes)]
    precision = [class_table[i][i] / (np.sum(class_table[:, i]) + e)
                 for i in range(num_classes)]
    fscore = [2.0 * precision[i] * recall[i] / (precision[i] + recall[i])
              for i in range(num_classes)]
    headers1 = [str(i) for i in range(num_classes)]
    headers2 = ['Class'] + [str(i) for i in range(num_classes)]
    row1 = ['Precision'] + ['%.2f' % (p * 100.0) for p in precision]
    row2 = ['Recall'] + ['%.2f' % (r * 100.0) for r in recall]
    row3 = ['F1-Score'] + ['%.2f' % (f * 100.0) for f in fscore]

    accu = accuracy(predictions, np.argmax(labels, 1))
    log = open(dirname + '/test.log', 'a')
    log.write('\n*****************    %s    *******************\n' % dataset)
    log.write('Accuracy: %.6f\n' % accu)
    log.write(tabulate(class_table, headers1) + '\n')
    log.write(tabulate([row1, row2, row3], headers2) + '\n')
    log.write('******************************************************\n')
    if print_hyper:
        log.write('\n*********** Hyperparameter Summary ***********\n')
        for (key, val) in hyperparameter.items():
            log.write('%s = %s\n' % (key, str(val)))

        log.write('***********************************************\n')

    log.close()

    log = open(dirname + '/test.log', 'r')
    print(log.read())
    log.close()

    return class_table


def LinearSVM(train_dataset, train_labels,
              test_dataset, test_labels):
    print('Use linear SVM')
    y = np.argmax(train_labels, 1)
    # class_weight = {0: 1, 1: 1, 2: 1, 3: 1000, 4: 1000}
    binary_class_weight = {0: 1, 1: 1}
    clf = svm.LinearSVC(class_weight=binary_class_weight, verbose=False)
    clf.fit(train_dataset, y)

    hyperparameter = {'class_weight': binary_class_weight,
                      'Linear': 'True'}
    train_prediction = clf.predict(train_dataset)
    compute_classification_table(train_prediction, train_labels,
                                 'LinearSVM', 'TRAIN SET')
    test_prediction = clf.predict(test_dataset)
    compute_classification_table(test_prediction, test_labels,
                                 'LinearSVM', 'TEST SET',
                                 hyperparameter, True)


def NonLinearSVM(train_dataset, train_labels,
                 test_dataset, test_labels):
    print('Use non-linear SVM')
    y = np.argmax(train_labels, 1)
    # class_weight = {0: 1, 1: 1, 2: 1, 3: 1000, 4: 1000}
    binary_class_weight = {0: 1, 1: 1}
    clf = svm.SVC(class_weight=binary_class_weight, verbose=False)
    clf.fit(train_dataset, y)

    hyperparameter = {'class_weight': binary_class_weight,
                      'Linear': 'False'}
    train_prediction = clf.predict(train_dataset)
    compute_classification_table(train_prediction, train_labels,
                                 'NonLinearSVM', 'TRAIN SET')
    test_prediction = clf.predict(test_dataset)
    compute_classification_table(test_prediction, test_labels,
                                 'NonLinearSVM', 'TEST SET',
                                 hyperparameter, True)


if __name__ == '__main__':
    train_dataset, train_labels, _, _, test_dataset, test_labels = \
        load_unsw_dataset()
    NonLinearSVM(train_dataset, train_labels, test_dataset, test_labels)
    LinearSVM(train_dataset, train_labels, test_dataset, test_labels)
