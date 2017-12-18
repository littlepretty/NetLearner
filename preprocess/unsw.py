from __future__ import print_function
import csv
from sets import Set
import numpy as np
import pandas
from sklearn.preprocessing import OneHotEncoder
from netlearner.utils import maybe_npsave


def generate_header(feature_names):
    header = ''
    for name in feature_names:
        header += name + ','

    return header


def discovery_feature_volcabulary(filenames, verbose=False):
    proto, state, service = Set(), Set(), Set()
    for filename in filenames:
        csv_file = open(filename, 'r')
        csv_file.readline()
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            proto.add(row[2])
            state.add(row[4])
            service.add(row[3])

    proto = list(proto)
    state = list(state)
    service = list(service)
    result = {'proto': proto, 'state': state, 'service': service}
    if verbose is True:
        for (key, value) in result.items():
            print(key, value, len(value))

    return result


def get_feature_names(filename):
    f = open(filename, 'r')
    f.readline()  # skip description line
    feature_names = []
    symbolic = []
    continuous = []
    discrete = []
    for line in f.readlines():
        contents = line.strip('\n').split(',')
        feature_names.append(contents[1])
        if contents[2] == 'nominal':
            symbolic.append(contents[1])
        elif contents[2] == 'float':
            continuous.append(contents[1])
        elif contents[2] == 'integer':
            discrete.append(contents[1])

    return feature_names, symbolic, continuous, discrete


def discovery_range(filenames, dnames, headers):
    max_list = {x: 0 for x in dnames}
    min_list = {x: 2424250010 for x in dnames}  # stime starts at 1.4 billion
    for filename in filenames:
        data_frame = pandas.read_csv(filename,
                                     sep=',',
                                     names=headers,
                                     engine='python',
                                     na_values='-',
                                     skiprows=1)
        for name in dnames:
            column = data_frame[name]
            max_list[name] = max(max_list[name], column.max(skipna=True))
            min_list[name] = min(min_list[name], column.min(skipna=True))

    small_ranges = []
    for name in dnames:
        # print('%s ranges: [%s, %s]' % (name, min_list[name], max_list[name]))
        if max_list[name] - min_list[name] < 12000:
            small_ranges.append(name)

    # print('features with small value range:', small_ranges)
    return max_list, min_list, small_ranges


def discovery_integer_map(feature_file, dataset_names):
    headers, _, _, dnames = get_feature_names(feature_file)
    max_list, min_list, _ = discovery_range(dataset_names,
                                            dnames, headers)
    result = {}
    for (name, maximum) in max_list.items():
        result[name] = {'min': min_list[name], 'max': max_list[name]}

    return result


def discovery_continuous_map(feature_file, dataset_names):
    headers, _, cnames, _ = get_feature_names(feature_file)
    max_list, min_list, _ = discovery_range(dataset_names,
                                            cnames, headers)
    result = {}
    for (name, maximum) in max_list.items():
        result[name] = {'min': min_list[name], 'max': max_list[name]}

    return result


def discovery_discrete_range(filenames, dnames, headers):
    return discovery_range(filenames, dnames, headers)


def discovery_category_map(filenames):
    proto, pint = dict(), 0
    service, sint = dict(), 0
    state, stint = dict(), 0
    attack, aint = dict(), 0
    for filename in filenames:
        csv_file = open(filename, 'rb')
        next(csv_file)
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            if row[2] not in proto:
                proto[row[2]] = pint
                pint += 1
            if row[3] not in service:
                service[row[3]] = sint
                sint += 1
            if row[4] not in state:
                state[row[4]] = stint
                stint += 1
            if row[-2] not in attack:
                attack[row[-2]] = aint
                aint += 1
    # print(proto)
    # print(service)
    # print(state)
    # print(attack)
    return {'proto': proto, 'service': service,
            'state': state, 'attack': attack}


def load_csv(filename, category_maps):
    print('Processing ' + filename)
    numerical_features = list()
    symbolic_features = list()
    labels = list()

    csv_file = open(filename, 'rb')
    next(csv_file)
    reader = csv.reader(csv_file, delimiter=',')
    for row in reader:
        row.pop(0)
        labels.append(row.pop())

        numerical_features.append(row[0:1] + row[4:-1])
        row[1] = category_maps['proto'][row[1]]
        row[2] = category_maps['service'][row[2]]
        row[3] = category_maps['state'][row[3]]
        symbolic_features.append(row[1:4])

    part1 = np.array(numerical_features, dtype=float)
    print('Numberic feature size:', part1.shape)

    part2 = np.array(symbolic_features, dtype=int)
    print('Symbolic feature size:', part2.shape)

    bincount = np.bincount(labels)
    print('Label distribution:', bincount)

    labels = np.array(labels, dtype=int)[np.newaxis]
    labels = labels.T
    print('Label size:', labels.shape)

    feature_max = np.amax(part2, axis=0)
    feature_min = np.amin(part2, axis=0)
    print('Max of integerized symbolic features', feature_max)
    print('Min of integerized symbolic features', feature_min)
    return part1, part2, labels


def encode_symbolic_feature(train_symbol, test_symbol):
    X = np.concatenate((train_symbol, test_symbol), axis=0)
    encoder = OneHotEncoder()
    encoder.fit(X)
    encoded_train = encoder.transform(train_symbol).toarray()
    encoded_test = encoder.transform(test_symbol).toarray()
    print('Symbolic feature size: ', encoded_train.shape, encoded_test.shape)
    print('One-Hot Encoder info: ', encoder.n_values_)

    return encoded_train, encoded_test


def encode_labels(labels, num_classes=2):
    encoded = np.zeros((labels.shape[0], num_classes), dtype=float)
    for (i, l) in enumerate(labels):
        encoded[i, int(l[0])] = 1.0
    return encoded


def get_indices_dist(labels):
    dist = []
    num_labels = labels.shape[1]
    for i in range(num_labels):
        example = np.zeros(shape=(num_labels))
        example[i] = 1
        indices = np.where(labels == example)[1]
        print(indices.shape)
        dist.append(indices)

    return dist


def split_valid(dataset, labels, percent=0.12):
    perm = np.random.permutation(dataset.shape[0])
    dataset = dataset[perm, :]
    labels = labels[perm, :]

    num_traffics = int(dataset.shape[0] * percent)
    valid_dataset = dataset[0: num_traffics, :]
    valid_labels = labels[0: num_traffics, :]

    # valid_dataset = np.ndarray(shape=(0, dataset.shape[1]))
    # valid_labels = np.ndarray(shape=(0, labels.shape[1]))
    # dist = get_indices_dist(labels)

    # for (i, indices) in enumerate(dist):
    # valid_indices = indices[0: num_traffics]
    # valid_dataset = np.concatenate((valid_dataset,
    # dataset[valid_indices, :]), axis=0)
    # valid_labels = np.concatenate((valid_labels,
    # labels[valid_indices, :]), axis=0)

    print('Valid dataset', valid_dataset.shape, valid_labels.shape)
    return valid_dataset, valid_labels


def generate_dataset(one_hot_encode=True, root_dir=''):
    prefix = 'UNSW/UNSW_NB15_'
    train_name = prefix + 'training-set.csv'
    test_name = prefix + 'testing-set.csv'
    category_maps = discovery_category_map([train_name, test_name])

    num_train, sym_train, train_labels = load_csv(train_name, category_maps)
    num_test, sym_test, test_labels = load_csv(test_name, category_maps)

    if one_hot_encode is True:
        sym_train, sym_test = encode_symbolic_feature(sym_train, sym_test)

    train_labels = encode_labels(train_labels)
    test_labels = encode_labels(test_labels)

    train_traffic = np.concatenate((num_train, sym_train), axis=1)
    test_traffic = np.concatenate((num_test, sym_test), axis=1)

    print('Trainset shape:', train_traffic.shape, train_labels.shape)
    maybe_npsave('%sUNSW/train_dataset' % root_dir, train_traffic)
    maybe_npsave('%sUNSW/train_labels' % root_dir, train_labels,
                 binary_label=True)

    valid_traffic, valid_labels = split_valid(test_traffic, test_labels)
    print('Validset shape:', valid_traffic.shape, valid_labels.shape)
    maybe_npsave('%sUNSW/valid_dataset' % root_dir, valid_traffic)
    maybe_npsave('%sUNSW/valid_labels' % root_dir, valid_labels,
                 binary_label=True)

    print('Testset shape:', test_traffic.shape, test_labels.shape)
    maybe_npsave('%sUNSW/test_dataset' % root_dir, test_traffic)
    maybe_npsave('%sUNSW/test_labels' % root_dir, test_labels,
                 binary_label=True)


if __name__ == '__main__':
    # generate_dataset(True)
    # generate_dataset(False)
    filenames = ['UNSW/UNSW_NB15_%s-set.csv' % x
                 for x in ['training', 'testing']]
    # discovery_category_map(filenames)
    filename = 'UNSW/feature_names_train_test.csv'
    discovery_feature_volcabulary(filenames)
    headers, _, _, dnames = get_feature_names(filename)
    discovery_discrete_range(filenames, dnames, headers)
