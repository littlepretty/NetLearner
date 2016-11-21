import numpy as np
from sklearn import svm
from netlearner.utils import standard_scale
from tabulate import tabulate


np.random.seed(5)

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


def accuracy(y, target):
    return float(np.sum(y == target)) / float(target.shape[0])


def compute_classification_table(predictions, labels):
    num_classes = labels.shape[1]
    class_table = np.zeros((num_classes, num_classes))
    actual_class = np.argmax(labels, 1)
    for (a, p) in zip(actual_class, predictions):
        class_table[a][p] += 1

    headers = [str(i) for i in range(labels.shape[1])]
    print(tabulate(class_table, headers))
    return class_table


y = np.argmax(train_labels, 1)
class_weight = {0: 1, 1: 1, 2: 1, 3: 1000, 4: 1000}
binary_class_weight = {0: 1, 1: 100}
clf = svm.SVC(class_weight=binary_class_weight, verbose=False)
print('Use non-linear SVM')
clf.fit(train_dataset, y)

train_prediction = clf.predict(train_dataset)
print('Trainset accuracy: %f' % accuracy(train_prediction, y))
compute_classification_table(train_prediction, train_labels)

test_prediction = clf.predict(test_dataset)
test_y = np.argmax(test_labels, 1)
print('Testset accuracy: %f' % accuracy(test_prediction, test_y))
compute_classification_table(test_prediction, test_labels)
