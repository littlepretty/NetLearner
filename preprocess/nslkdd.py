# Import python packages

from __future__ import print_function
import csv
import os
import sys
import numpy as np
import pandas
from sklearn.preprocessing import OneHotEncoder
# import argparse

# ### Convert/Define Symbolic String to Int
#
# Read in 1 csv file. There are in total 41 features for one traffic,
# 4 of which needs to be convert from symbolic to numeric.
# 1. protocol types
# 2. services
# 3. flag
# 4. attack

raw_feature_size = 41
# NOTE: should use consistent maps defined below
protocol_types = {'udp': 1, 'icmp': 2, 'tcp': 0}
service_types = {'urp_i': 11, 'netbios_ssn': 51, 'Z39_50': 40, 'tim_i': 68,
                 'smtp': 1, 'domain': 22, 'private': 12, 'echo': 18,
                 'printer': 50, 'red_i': 69, 'eco_i': 6, 'sunrpc': 43,
                 'ftp_data': 14, 'urh_i': 62, 'pm_dump': 38, 'pop_3': 13,
                 'pop_2': 52, 'systat': 30, 'ftp': 7, 'uucp': 37, 'whois': 21,
                 'tftp_u': 66, 'netbios_dgm': 41, 'efs': 54, 'remote_job': 25,
                 'sql_net': 57, 'daytime': 16, 'ntp_u': 8, 'finger': 4,
                 'ldap': 42, 'netbios_ns': 60, 'kshell': 61, 'iso_tsap': 59,
                 'ecr_i': 9, 'nntp': 36, 'http_2784': 63, 'shell': 33,
                 'domain_u': 2, 'uucp_path': 56, 'courier': 44, 'exec': 45,
                 'aol': 65, 'netstat': 15, 'telnet': 5, 'gopher': 24,
                 'rje': 26, 'hostnames': 55, 'link': 29, 'ssh': 17,
                 'http_443': 48, 'csnet_ns': 47, 'X11': 32, 'IRC': 39,
                 'harvest': 64, 'imap4': 35, 'icmp': 70, 'supdup': 28,
                 'name': 20, 'nnsp': 53, 'mtp': 23, 'http': 0, 'bgp': 46,
                 'ctf': 27, 'klogin': 49, 'vmnet': 58, 'time': 19,
                 'discard': 31, 'login': 34, 'auth': 3, 'other': 10,
                 'http_8001': 67}
flag_types = {'OTH': 4, 'RSTR': 8, 'S3': 3, 'S2': 1, 'S1': 2, 'S0': 7,
              'RSTOS0': 9, 'REJ': 5, 'SH': 10, 'RSTO': 6, 'SF': 0}
land_types = {'1': 1, '0': 0}
login_types = {'1': 0, '0': 1}
host_login_types = {'1': 1, '0': 0}
guest_login_types = {'1': 1, '0': 0}
attack_map = {'guess_passwd': 6, 'spy': 21, 'named': 24, 'ftp_write': 12,
              'processtable': 33, 'nmap': 17, 'back': 13, 'multihop': 18,
              'rootkit': 22, 'udpstorm': 30, 'snmpguess': 39, 'pod': 7,
              'apache2': 29, 'sqlattack': 38, 'portsweep': 9, 'ps': 34,
              'httptunnel': 35, 'sendmail': 27, 'snmpgetattack': 23,
              'perl': 3, 'ipsweep': 10, 'teardrop': 8, 'satan': 15,
              'loadmodule': 2, 'buffer_overflow': 1, 'mailbomb': 37,
              'mscan': 32, 'saint': 28, 'normal': 0, 'xterm': 31, 'phf': 16,
              'warezmaster': 19, 'imap': 14, 'warezclient': 20, 'land': 11,
              'neptune': 4, 'worm': 36, 'xlock': 25, 'smurf': 5, 'xsnoop': 26}


# ### Convert Attack to Int
# Process other data files according to the maps defined above.
# There are total 23 types of attacks. But we will further map these attacks
# to 4 categories, or even just a binary(attack, nonattack).
attack_category_map = {'normal': 'normal', 'back': 'dos',
                       'buffer_overflow': 'u2r',
                       'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
                       'ipsweep': 'probe', 'land': 'dos', 'loadmodule': 'u2r',
                       'multihop': 'r2l', 'neptune': 'dos', 'nmap': 'probe',
                       'perl': 'u2r', 'phf': 'r2l', 'pod': 'dos',
                       'portsweep': 'probe', 'rootkit': 'u2r', 'satan': 'probe',
                       'smurf': 'dos', 'spy': 'r2l', 'teardrop': 'dos',
                       'warezclient': 'r2l', 'warezmaster': 'r2l',
                       'snmpgetattack': 'dos',
                       'apache2': 'dos',
                       'arppoison': 'dos',
                       'back': 'dos',
                       'crashiis': 'dos',
                       'dosnuke': 'dos',
                       'land': 'dos',
                       'mailbomb': 'dos',
                       'syn flood': 'dos',
                       'neptune': 'dos',
                       'ping of death': 'dos',
                       'pod': 'dos',
                       'processtable': 'dos',
                       'selfping': 'dos',
                       'smurf': 'dos',
                       'sshprocesstable': 'dos',
                       'syslogd': 'dos',
                       'tcpreset': 'dos',
                       'teardrop': 'dos',
                       'udpstorm': 'dos',
                       'anypw': 'u2r',
                       'casesen': 'u2r',
                       'eject': 'u2r',
                       'ffbconfig': 'u2r',
                       'fdformat': 'u2r',
                       'loadmodule': 'u2r',
                       'ntfsdos': 'u2r',
                       'perl': 'u2r',
                       'ps': 'u2r',
                       'sechole': 'u2r',
                       'xterm': 'u2r',
                       'yaga': 'u2r',
                       'snmpguess': 'u2r',
                       'dictionary': 'r2l',
                       'ftpwrite': 'r2l',
                       'guest': 'r2l',
                       'httptunnel': 'r2l',
                       'imap': 'r2l',
                       'named': 'r2l',
                       'ncftp': 'r2l',
                       'netbus': 'r2l',
                       'netcat': 'r2l',
                       'phf': 'r2l',
                       'ppmacro': 'r2l',
                       'sendmail': 'r2l',
                       'sshtrojan': 'r2l',
                       'xlock': 'r2l',
                       'xsnoop': 'r2l',
                       'insidesniffer': 'probe',
                       'ipsweep': 'probe',
                       'ls_domain': 'probe',
                       'mscan': 'probe',
                       'ntinfoscan': 'probe',
                       'nmap': 'probe',
                       'queso': 'probe',
                       'resetscan': 'probe',
                       'saint': 'probe',
                       'satan': 'probe',
                       'secret': 'data',
                       'sqlattack': 'probe',
                       'worm': 'probe'}
# If category_map[x] != 0, then x is a type of attack
label_names = ['normal', 'probe', 'dos', 'u2r', 'r2l']
category_map = {'normal': 0, 'probe': 1, 'dos': 2, 'u2r': 3, 'r2l': 4}
binary_map = {'normal': 0, 'probe': 1, 'dos': 1, 'u2r': 1, 'r2l': 1, 'other': 1}


def load_traffic(filename, encoder_fitted, traffic_map=category_map,
                 one_hot_encode=True, show=6):
    """Each row  of all_traffic is a traffic record"""
    numerical_features = list()
    symbolic_features = list()
    labels = list()

    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        seen = set()
        header = True
        for row in reader:
            if header:
                header = False  # Skip feature names/header
                continue
            try:
                # Ignore difficulty level
                row = row[:-1]
                attack = row[-1]
                row = row[0:-1]
                row[1] = protocol_types[row[1]]
                row[2] = service_types[row[2]]
                row[3] = flag_types[row[3]]
                row[6] = land_types[row[6]]
                row[11] = login_types[row[11]]
                row[20] = host_login_types[row[20]]
                row[21] = guest_login_types[row[21]]
                # Ignore the 19th feature, which is a constant zero
                numerical_features.append(row[0:1] + row[4:6] + row[7:11] +
                                          row[12:19] + row[22:41])
                symbolic_features.append(row[1:4] + row[6:7] + row[11:12] +
                                         row[20:22])

                category = attack_category_map[attack]
                labels.append(traffic_map[category])

                if category not in seen and show > 0:
                    print(category, ' traffic')
                    show -= 1
                    seen.add(category)
            except KeyError as e:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                print('Cannot parse %s at line %d' % (e, exc_tb.tb_lineno))

    part1 = np.array(numerical_features, dtype=float)
    print('Numeric feature size: ', part1.shape[1])

    if one_hot_encoding is True:
        if encoder_fitted is False:
            enc = OneHotEncoder()
            enc.fit(symbolic_features)

        encoded = enc.transform(symbolic_features).toarray()
        print('One-Hot Encoded symbolic features: ', encoded.shape)
        part2 = np.array(encoded, dtype=float)
        print('Symbolic feature sizes: ', enc.n_values_, part2.shape)
    else:
        part2 = np.array(symbolic_features, dtype=float)
        print('Symbolic feature sizes: ', part2.shape)

    all_traffics = np.concatenate((part1, part2), axis=1)
    labels = np.array(labels, dtype=int)[np.newaxis]
    dist = []
    for (i, name) in enumerate(label_names):
        print('#%s = %d' % (name,  np.sum(labels == i)))
        dist.append(np.where(labels == i)[1])

    labels = labels.T
    all_traffics = np.concatenate((all_traffics, labels), axis=1)

    print('Traffic data with label: ', all_traffics.shape)
    return all_traffics, dist


def one_hot_encoding(labels):
    """Given labels which is a N dimentioanl vector,
    return a N by C matrix where each row is one-hot-encoding of each label"""
    encoding = np.zeros((labels.shape[0], num_classes), dtype=float)
    for [i, l] in enumerate(labels):
        encoding[i, int(l)] = 1.0
    return encoding


def shuffle_dataset_with_label(matrix, contain_label=True):
    """If we are doing supervised learning, dataset should
    contain labels s.t. we shuffle data along with labels"""
    # matrix size is N by F
    # N = #records and F = #features or +1 if containing label
    permutation = np.random.permutation(matrix.shape[0])
    matrix = matrix[permutation, :]
    if contain_label:
        dataset = matrix[:, :-1]
        labels = matrix[:, -1]
        print('Convert label to one-hot-encoding...')
        labels = one_hot_encoding(labels)
        return dataset, labels
    else:
        return matrix, None


def split_dataset_with_label(matrix):
    # matrix size is N by F
    # N = #records and F = #features +1 because it containing label
    dataset = matrix[:, :-1]
    labels = matrix[:, -1]
    print('Convert label to one-hot-encoding...')
    labels = one_hot_encoding(labels)
    return dataset, labels


def maybe_npsave(dataname, data, force=True):
    filename = dataname + '.npy'
    if os.path.exists(filename) and not force:
        print('%s already exists - Skip saving.' % filename)
    else:
        print('Writing %s to %s...' % (dataname, filename))
        np.save(filename, data)
        print('Finish saving ', dataname)
    return filename


def generate_train_dataset(dataset, labels, root=''):
    maybe_npsave(root + 'NSLKDD/train_dataset', dataset)
    maybe_npsave(root + 'NSLKDD/train_labels', labels)

    print('Training', dataset.shape, labels.shape)


def generate_valid_test_dataset(dataset, labels, dist, percent=0.1, root=''):
    print('Original Test dataset ', dataset.shape, labels.shape)
    valid_dataset = np.ndarray(shape=(0, dataset.shape[1]))
    valid_label = np.ndarray(shape=(0, labels.shape[1]))

    for (i, indices) in enumerate(dist):
        print('Traffic %s in dataset: %s' % (label_names[i],
                                             dataset[indices, :].shape))
        num_traffics = int(len(indices) * percent)
        shuffled_indices = np.random.permutation(indices)
        valid_indices = shuffled_indices[0: num_traffics]
        valid_dataset = np.concatenate(
            (valid_dataset, dataset[valid_indices, :]),
            axis=0)
        valid_label = np.concatenate((valid_label,
                                      labels[valid_indices, :]),
                                     axis=0)

    print('Test dataset ', dataset.shape, labels.shape)
    maybe_npsave(root + 'NSLKDD/test_dataset', dataset)
    maybe_npsave(root + 'NSLKDD/test_labels', labels)

    print('Valid dataset ', valid_dataset.shape, valid_label.shape)
    maybe_npsave(root + 'NSLKDD/valid_dataset', valid_dataset)
    maybe_npsave(root + 'NSLKDD/valid_labels', valid_label)


def generate_datasets(binary_label, one_hot_encoding=False, root=''):
    global num_classes
    train = 'NSLKDD/KDDTrain.csv'
    if binary_label:
        print('Use binary label')
        num_classes = 2
        data_matrix, dist = load_traffic(train, False,
                                         binary_map, one_hot_encoding)
    else:
        num_classes = len(category_map)
        data_matrix, dist = load_traffic(train, False, one_hot_encoding)

    dataset, labels = shuffle_dataset_with_label(data_matrix)
    generate_train_dataset(dataset, labels, root)

    test = 'NSLKDD/KDDTest.csv'
    if binary_label:
        print('Use binary label')
        num_classes = 2
        data_matrix, dist = load_traffic(test, True,
                                         binary_map, one_hot_encoding)
    else:
        num_classes = len(category_map)
        data_matrix, dist = load_traffic(test, True, one_hot_encoding)

    dataset, labels = split_dataset_with_label(data_matrix)
    generate_valid_test_dataset(dataset, labels, dist, root=root)


def get_feature_names(feature_file):
    f = open(feature_file, 'r')
    f.readline()  # skip column head
    feature_names = []
    symbolic = []
    discrete = []
    continuous = []
    for line in f.readlines():
        contents = line.strip('\n').split(' ')
        feature_names.append(contents[1])
        if contents[2] == 'symbolic':
            symbolic.append(contents[1])
        elif contents[2] == 'continuous':
            continuous.append(contents[1])
        elif contents[2] == 'integer':
            discrete.append(contents[1])

    return feature_names, symbolic, continuous, discrete


def get_categorical_values(name):
    feature = None
    if name == 'protocol':
        feature = protocol_types
    elif name == 'service':
        feature = service_types
    elif name == 'flag':
        feature = flag_types
    elif name == 'land':
        feature = land_types
    elif name == 'login':
        feature = login_types
    elif name == 'host_login':
        feature = host_login_types
    elif name == 'guest_login':
        feature = guest_login_types

    return feature.keys() if feature is not None else None


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

    return max_list, min_list


def discovery_integer_map(feature_file, dataset_names):
    headers, _, _, dnames = get_feature_names(feature_file)
    max_list, min_list = discovery_range(dataset_names,
                                         dnames, headers)
    result = {}
    for (name, maximum) in max_list.items():
        result[name] = {'min': min_list[name], 'max': max_list[name]}

    return result


def discovery_continuous_map(feature_file, dataset_names):
    headers, _, cnames, _ = get_feature_names(feature_file)
    max_list, min_list = discovery_range(dataset_names,
                                         cnames, headers)
    result = {}
    for (name, maximum) in max_list.items():
        result[name] = {'min': min_list[name], 'max': max_list[name]}

    return result


def discovery_feature_volcabulary(filenames, verbose=True):
    features = dict()
    features['protocol'] = protocol_types.keys()
    features['service'] = service_types.keys()
    features['flag'] = flag_types.keys()
    features['land'] = land_types.keys()
    features['login'] = login_types.keys()
    features['host_login'] = host_login_types.keys()
    features['guest_login'] = guest_login_types.keys()
    return features


if __name__ == '__main__':
    np.set_printoptions(precision=4)
    binary_label = False
    generate_datasets(binary_label)
    # generate_datasets()
