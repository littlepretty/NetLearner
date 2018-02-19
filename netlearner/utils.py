from __future__ import print_function, division
import numpy as np
import os
import errno
import tensorflow as tf
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.preprocessing import FunctionTransformer
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math


def min_max_scale(X_train, X_valid, X_test):
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(X_train)
    norm_train = scaler.transform(X_train)
    norm_valid = scaler.transform(X_valid) if X_valid is not None else None
    norm_test = scaler.transform(X_test) if X_test is not None else None
    return norm_train, norm_valid, norm_test


def standard_scale(X_train, X_valid, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid) if X_valid is not None else None
    X_test = scaler.transform(X_test) if X_test is not None else None
    return X_train, X_valid, X_test


def interquartile_scale(X_train, X_valid, X_test):
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid) if X_valid is not None else None
    X_test = scaler.transform(X_test) if X_test is not None else None
    return X_train, X_valid, X_test


def quantile_transform(X_train, X_valid, X_test, columns):
    t = QuantileTransformer()
    t.fit(X_train[:, columns])
    qX_train = t.transform(X_train[:, columns])
    qX_valid = t.transform(X_valid[:, columns]) \
        if X_valid is not None else None
    qX_test = t.transform(X_test[:, columns]) if X_test is not None else None
    if X_valid is not None:
        X_train[:, columns] = qX_train
        X_valid[:, columns] = qX_valid
        X_test[:, columns] = qX_test
        return X_train, X_valid, X_test
    else:
        return X_train


def augment_quantiled(X_train, X_valid, X_test, columns):
    t = QuantileTransformer()
    t.fit(X_train[:, columns])
    qX_train = t.transform(X_train[:, columns])
    qX_valid = t.transform(X_valid[:, columns]) \
        if X_valid is not None else None
    qX_test = t.transform(X_test[:, columns]) if X_test is not None else None
    mX_train, mX_valid, mX_test = min_max_scale(X_train, X_valid, X_test)
    X_train = np.concatenate((mX_train, qX_train), axis=1)
    if qX_valid is None:
        return X_train
    else:
        X_valid = np.concatenate((mX_valid, qX_valid), axis=1)
        X_test = np.concatenate((mX_test, qX_test), axis=1)
        return X_train, X_valid, X_test


def log_transform(X_train, X_valid, X_test, columns):
    t = FunctionTransformer(np.log1p)
    part_X_train = t.transform(X_train[:, columns])
    part_X_train = t.transform(X_train[:, columns])
    part_X_valid = t.transform(X_valid[:, columns])
    part_X_test = t.transform(X_test[:, columns])
    X_train[:, columns] = part_X_train
    X_valid[:, columns] = part_X_valid
    X_test[:, columns] = part_X_test
    return X_train, X_valid, X_test


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) ==
                          np.argmax(labels, 1)) / predictions.shape[0]


def accuracy_binary(predictions, labels):
    predicted_class = np.argmax(predictions, 1)
    actual_class = np.argmax(labels, 1)
    correct1 = np.logical_and(np.greater(predicted_class, 0),
                              np.greater(actual_class, 0))
    correct2 = np.logical_and(predicted_class == 0, actual_class == 0)
    return 100.0 * (np.sum(correct1) + np.sum(correct2)) / predictions.shape[0]


def compute_classification_table(predictions, labels):
    n_cls = labels.shape[1]
    class_table = np.zeros((n_cls, n_cls))
    predicted_cls = np.argmax(predictions, 1)
    actual_cls = np.argmax(labels, 1)

    for (a, p) in zip(actual_cls, predicted_cls):
        class_table[a][p] += 1

    return class_table


def compute_classification_table_binary(predictions, labels):
    class_table = np.zeros((2, 2))
    predicted_class = np.argmax(predictions, 1)
    actual_class = np.argmax(labels, 1)
    for (a, p) in zip(actual_class, predicted_class):
        class_table[int(a > 0)][int(p > 0)] += 1

    return class_table


def correct_percentage(matrix, dataset_name='Test'):
    """
    :param matrix: map from actual to predicted
    :return: precision and recall measurement
    """
    epsilon = 1e-26
    num_classes = matrix.shape[0]
    weights = np.array([np.sum(matrix[i, :]) / np.sum(matrix) for i
                        in range(num_classes)])
    weights = np.reshape(weights, [num_classes, 1])

    recall = np.array([matrix[i][i] / (np.sum(matrix[i, :]) + epsilon) for i
                       in range(num_classes)])
    avg_recall = np.dot(recall, weights)
    recall = np.append(recall, avg_recall)

    precision = np.array([matrix[i][i] / (np.sum(matrix[:, i]) + epsilon)
                          for i in range(num_classes)])
    avg_precision = np.dot(precision, weights)
    precision = np.append(precision, avg_precision)

    fscore = np.array([2 * precision[i] * recall[i] / (precision[i] +
                                                       recall[i] + epsilon)
                       for i in range(num_classes)])
    avg_fscore = np.dot(fscore, weights)
    fscore = np.append(fscore, avg_fscore)

    headers = ['Class'] + [str(i) for i in range(matrix.shape[0])] + \
        ['Wtd. Avg.']
    row1 = ['Precision'] + ['%.2f' % (p * 100.0) for p in precision]
    row2 = ['Recall'] + ['%.2f' % (r * 100.0) for r in recall]
    row3 = ['F1-Score'] + ['%.2f' % (f * 100.0) for f in fscore]
    if dataset_name == 'Test':
        return tabulate([row1, row2, row3], headers) + '\n' + \
               tabulate([row1, row2, row3], headers, tablefmt='latex')
    else:
        return tabulate([row1, row2, row3], headers)


def xavier_init(fan_in, fan_out, constant=1):
    xavier_std = 2.0 / (fan_in + fan_out)
    return tf.truncated_normal((fan_in, fan_out), mean=0.0,
                               stddev=xavier_std, dtype=tf.float64)
    # low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    # high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    # return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high,
    # dtype=tf.float64)


def sample_prob_dist(prob, rand):
    return tf.nn.relu(tf.sign(prob - rand))


def measure_prediction(predictions, labels, dirname, dataset_name='Test'):
    log = open(dirname + 'confusion_matrix.log', 'a')
    if labels.shape[1] > 2:
        log.write("***** %d-Class performance *****\n" % labels.shape[1])
        accu = accuracy(predictions, labels)
        log.write("%sset accuracy: %f%%\n" % (dataset_name, accu))
        headers = [str(i) for i in range(labels.shape[1])]
        class_table = compute_classification_table(predictions, labels)
        log.write(tabulate(class_table, headers) + '\n')
        if dataset_name == 'Test':
            log.write(tabulate(class_table, headers, tablefmt='latex') + '\n')
        log.write(correct_percentage(class_table) + '\n')
        log.close()
        return class_table

    elif labels.shape[1] == 2:
        log.write("***** 2-Class performance *****\n")
        accu_binary = accuracy_binary(predictions, labels)
        log.write("%sset accuracy: %f%%\n" % (dataset_name, accu_binary))
        binary_headers = [str(i) for i in [0, 1]]
        binary_class_table = compute_classification_table_binary(predictions,
                                                                 labels)
        log.write(tabulate(binary_class_table, binary_headers) + '\n')
        if dataset_name == 'Test':
            log.write(tabulate(binary_class_table, binary_headers,
                               tablefmt='latex') + '\n')
        log.write(correct_percentage(binary_class_table, dataset_name) + '\n')
        log.close()
        return binary_class_table


def hyperparameter_summary(dirname, hyperparameter):
    f = open(dirname + '/test.log', 'a')
    f.write('\n******* Hyperparameter Summary *******\n')
    for (key, val) in hyperparameter.items():
        f.write('%s = %s\n' % (key, str(val)))

    f.write('**************************************\n')
    f.close()


def maybe_npsave(dataname, data, force=True, binary_label=False):
    note = ""
    if binary_label:
        note = ' as binary lables'
    filename = dataname + '.npy'
    if os.path.exists(filename) and not force:
        print('%s already exists - Skip saving.' % filename)
    else:
        np.save(filename, data)
        print('Finish saving %s to %s%s' % (dataname, filename, note))
    return filename


def maybe_npsave_range(dataname, data, l, r, force=False, binary_label=False):
    if binary_label:
        dataname += '_bin'
    filename = dataname + '.npy'
    if os.path.exists(filename) and not force:
        print('%s already exists - Skip saving.' % filename)
    else:
        save_data = data[l:r, :]
        print('Writing %s to %s...' % (dataname, filename))
        np.save(filename, save_data)
        print('Finish saving %s to %s' % (dataname, filename))
    return filename


def get_batch(train_dataset, train_labels, step, batch_size):
    offset = int(batch_size * step) % train_labels.shape[0]
    end = int(offset + batch_size) % train_labels.shape[0]
    if end < offset:
        batch_data = np.concatenate((train_dataset[offset:, :],
                                     train_dataset[:end, :]), axis=0)
        batch_labels = np.concatenate((train_labels[offset:, :],
                                       train_labels[:end, :]), axis=0)
    else:
        batch_data = train_dataset[offset:int(offset + batch_size), :]
        batch_labels = train_labels[offset:int(offset + batch_size), :]

    # print(batch_data.shape)
    # print(batch_labels.shape)
    return batch_data, batch_labels


def get_random_batch(dataset, labels, batch_size):
    num_rows = dataset.shape[0]
    indices = np.random.choice(range(num_rows), batch_size, False)
    return dataset[indices, :], labels[indices, :]

def next_batch(dataset, step, batch_size):
    offset = int(batch_size * step) % dataset.shape[0]
    end = int(offset + batch_size) % dataset.shape[0]
    if end < offset:
        return np.concatenate((dataset[offset:, :], dataset[0:end, :]), axis=0)
    else:
        return dataset[offset:end, :]


def create_dir(dirname):
    try:
        os.mkdir(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def attach_candidate_labels(dataset, num_labels):
    candidates = np.zeros([num_labels, dataset.shape[0], num_labels])
    result = np.zeros([num_labels, dataset.shape[0],
                       dataset.shape[1] + num_labels])
    for i in range(num_labels):
        candidates[i, :, i] = np.ones((dataset.shape[0]))
        result[i] = np.concatenate((dataset, candidates[i]), axis=1)
    return result


def plot_samples(samples, dirname, fig_index, name='sample'):
    num_samples, feature_dim = samples.shape
    size = int(math.sqrt(num_samples))
    image = int(math.sqrt(feature_dim))
    if (size * size != num_samples) or (image * image != feature_dim):
        return

    fig = plt.figure(figsize=(size, size))
    gs = gridspec.GridSpec(size, size, wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(image, image), cmap='Greys_r',
                   interpolation='none')

    plt.savefig('%s/%s_%d.png' % (dirname, name, fig_index),
                bbox_inches='tight', format='png')
    plt.close(fig)


def plot_V(dirname, D_V, G_V):
    fig, ax = plt.subplots()
    ax.plot(range(len(D_V)), D_V, 'rs-', label='V(D)')
    ax.plot(range(len(G_V)), G_V, 'bd-', label='V(G)')
    ax.legend(loc='upper right')
    plt.grid()
    plt.savefig('%s/compare_DG.png' % dirname, bbox_inches='tight')
    plt.close(fig)


def permutate_dataset(dataset, labels, name='Training'):
    perm = np.random.permutation(dataset.shape[0])
    perm_dataset = dataset[perm, :]
    perm_labels = labels[perm, :]
    print('%s set' % name, perm_dataset.shape, perm_labels.shape)
    return perm_dataset, perm_labels


def plot_traffic_as_image(dataset, labels, signiture,
                          name, num_samples, dirname='UNSW'):
    index = np.where(np.all(labels == signiture, axis=1))[0]
    matches = dataset[index, :]
    print('Real %s set' % name, matches.shape)
    sample_index = np.random.choice(matches.shape[0],
                                    num_samples, replace=False)
    samples = matches[sample_index, :]
    plot_samples(samples, dirname, num_samples, name)
