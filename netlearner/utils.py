from __future__ import print_function
import numpy as np
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
                           np.argmax(labels, 1)) / predictions.shape[0])


def compute_classification_table(predictions, labels):
    num_classes = labels.shape[1]
    class_table = np.zeros((num_classes, num_classes))
    predicted_class = np.argmax(predictions, 1)
    actual_class = np.argmax(labels, 1)
    for (a, p) in zip(actual_class, predicted_class):
        class_table[a][p] += 1

    return class_table


def correct_percentage(matrix):
    epsilon = 1e-20
    num_classes = matrix.shape[0]
    act2pred = [matrix[i][i] / (np.sum(matrix[i, :]) + epsilon)
                for i in range(num_classes)]
    pred2act = [matrix[i][i] / (np.sum(matrix[:, i]) + epsilon)
                for i in range(num_classes)]
    print(act2pred)
    print(pred2act)


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


def sample_prob_dist(prob, rand):
    return tf.nn.relu(tf.sign(prob - rand))


def measure_prediction(predictions, labels, dataset_name='Test'):
    accu = accuracy(predictions, labels)
    print("%sset accuracy: %f%%" % (dataset_name, accu))
    headers = [str(i) for i in range(labels.shape[1])]
    class_table = compute_classification_table(predictions, labels)
    print(tabulate(class_table, headers))
    correct_percentage(class_table)
