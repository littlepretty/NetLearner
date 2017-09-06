import numpy as np
from sklearn import svm
from svm import compute_classification_table

train_dataset = np.load('trainset.rbm.npy')
train_labels = np.load('UNSW/train_labels.npy')
test_dataset = np.load('testset.rbm.npy')
test_labels = np.load('UNSW/test_labels.npy')
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
weight = {0: 1, 1: 1, 2: 1, 3: 1000, 4: 1000}
binary_weight = {0: 1, 1: 100}


def LinearSVM(train_dataset, train_labels, test_dataset, test_labels):
    y = np.argmax(train_labels, 1)
    clf = svm.LinearSVC(class_weight=binary_weight, verbose=False)
    print('Use linear SVM')
    clf.fit(train_dataset, y)

    train_prediction = clf.predict(train_dataset)
    compute_classification_table(train_prediction, train_labels,
                                 'LinearSVM', 'RBM-Encoded TRAIN SET')
    test_prediction = clf.predict(test_dataset)
    compute_classification_table(test_prediction, test_labels,
                                 'LinearSVM', 'RBM-Encoded TEST SET')


def NonLinearSVM(train_dataset, train_labels, test_dataset, test_labels):
    y = np.argmax(train_labels, 1)
    clf = svm.SVC(class_weight=binary_weight, verbose=False)
    print('Use non-linear SVM')
    clf.fit(train_dataset, y)

    train_prediction = clf.predict(train_dataset)
    compute_classification_table(train_prediction, train_labels,
                                 'NonLinearSVM', 'RBM-Encoded TRAIN SET')
    test_prediction = clf.predict(test_dataset)
    compute_classification_table(test_prediction, test_labels,
                                 'NonLinearSVM', 'RBM-Encoded TEST SET')


if __name__ == '__main__':
    NonLinearSVM(train_dataset, train_labels, test_dataset, test_labels)
    LinearSVM(train_dataset, train_labels, test_dataset, test_labels)
