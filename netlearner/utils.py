from __future__ import print_function
import numpy as np
import os
import errno
import tensorflow as tf
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def min_max_normalize(X_train, X_valid, X_test):
    preprocessor = MinMaxScaler(feature_range=(0, 1)).fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)
    return X_train, X_valid, X_test


def standard_scale(X_train, X_valid, X_test):
    preprocessor = StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)
    return X_train, X_valid, X_test


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) ==
                           np.argmax(labels, 1)) / float(predictions.shape[0]))


def accuracy_binary(predictions, labels):
    predicted_class = np.argmax(predictions, 1)
    actual_class = np.argmax(labels, 1)
    correct1 = np.logical_and(np.greater(predicted_class, 0), np.greater(actual_class, 0))
    correct2 = np.logical_and(predicted_class == 0, actual_class == 0)
    return 100.0 * (np.sum(correct1) + np.sum(correct2)) / float(predictions.shape[0])


def compute_classification_table(predictions, labels):
    num_classes = labels.shape[1]
    class_table = np.zeros((num_classes, num_classes))
    predicted_class = np.argmax(predictions, 1)
    actual_class = np.argmax(labels, 1)
    for (a, p) in zip(actual_class, predicted_class):
        class_table[a][p] += 1

    return class_table


def compute_classification_table_binary(predictions, labels):
    class_table = np.zeros((2, 2))
    predicted_class = np.argmax(predictions, 1)
    actual_class = np.argmax(labels, 1)
    for (a, p) in zip(actual_class, predicted_class):
        class_table[int(a > 0)][int(p > 0)] += 1

    return class_table


def correct_percentage(matrix):
    """
    :param matrix: map from actual to predicted
    :return: precision and recall measurement
    """
    epsilon = 1e-26
    num_classes = matrix.shape[0]
    precision = [matrix[i][i] / (np.sum(matrix[i, :]) + epsilon)
                 for i in range(num_classes)]
    recall = [matrix[i][i] / (np.sum(matrix[:, i]) + epsilon)
              for i in range(num_classes)]
    fscore = [2 * precision[i] * recall[i] / (precision[i] + recall[i] + epsilon)
              for i in range(len(precision))]

    headers = ['Class'] + [str(i) for i in range(matrix.shape[0])]
    row1 = ['Precision'] + ['%.2f%%' % (p * 100.0) for p in precision]
    row2 = ['Recall'] + ['%.2f%%' % (r * 100.0) for r in recall]
    row3 = ['F1-Score'] + ['%.2f%%' % (f * 100.0) for f in fscore]
    print(tabulate([row1, row2, row3], headers))


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def sample_prob_dist(prob, rand):
    return tf.nn.relu(tf.sign(prob - rand))


def measure_prediction(predictions, labels, dataset_name='Test'):
    print("***** 5-Class performance *****")
    accu = accuracy(predictions, labels)
    print("%sset accuracy: %f%%" % (dataset_name, accu))
    headers = [str(i) for i in range(labels.shape[1])]
    class_table = compute_classification_table(predictions, labels)
    print(tabulate(class_table, headers))
    correct_percentage(class_table)

    print("***** 2-Class performance *****")
    accu_binary = accuracy_binary(predictions, labels)
    print("%sset accuracy: %f%%" % (dataset_name, accu_binary))
    binary_headers = [str(i) for i in [0, 1]]
    binary_class_table = compute_classification_table_binary(predictions, labels)
    print(tabulate(binary_class_table, binary_headers))
    correct_percentage(binary_class_table)


def maybe_npsave(dataname, data, l, r, force=False, binary_label=False):
    if binary_label:
        dataname = dataname + '_bin'
    filename = dataname + '.npy'
    if os.path.exists(filename) and not force:
        print('%s already exists - Skip saving.' % filename)
    else:
        save_data = data[l:r, :]
        print('Writing %s to %s...' % (dataname, filename))
        np.save(filename, save_data)
        print('Finish saving ', dataname)
    return filename


def get_batch(train_dataset, train_labels, step, batch_size):
    offset = (batch_size * step) % train_labels.shape[0]
    end = (offset + batch_size) % train_labels.shape[0]
    if end < offset:
        batch_data = np.concatenate((train_dataset[offset:, :],
                                     train_dataset[:end, :]), axis=0)
        batch_labels = np.concatenate((train_labels[offset:, :],
                                       train_labels[:end, :]), axis=0)
    else:
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

    # print(batch_data.shape)
    # print(batch_labels.shape)
    return batch_data, batch_labels


def create_dir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
