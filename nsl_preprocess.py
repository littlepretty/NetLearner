# Import python packages

from __future__ import print_function
import csv
import os
import numpy as np
# from sklearn import preprocessing
# import argparse

# ### Convert/Define Symbolic String to Int
#
# Read in 1 csv file. There are in total 41 features for one traffic,
# 4 of which needs to be convert from symbolic to numeric.
# 1. protocol types
# 2. services
# 3. flag
# 4. attack


def load_feature_maps(files=['traindata', 'testdata']):
    protocol_map = {}
    service_map = {}
    flag_map = {}
    attack_map = {}
    for datafile in files:
        filename = 'data/kddcup.' + datafile
        with open(filename, 'rb') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                if row[1] not in protocol_map:
                    protocol_map[row[1]] = len(protocol_map)
                if row[2] not in service_map:
                    service_map[row[2]] = len(service_map)
                if row[3] not in flag_map:
                    flag_map[row[3]] = len(flag_map)
                if row[-1][:-1] not in attack_map:
                    attack_map[row[-1][:-1]] = len(attack_map)
    return protocol_map, service_map, flag_map, attack_map


feature_size = 41
# NOTE: should use consistent maps defined below
protocol_map = {'udp': 1, 'icmp': 2, 'tcp': 0}
service_map = {'urp_i': 11, 'netbios_ssn': 51, 'Z39_50': 40, 'tim_i': 68,
               'smtp': 1, 'domain': 22, 'private': 12, 'echo': 18,
               'printer': 50, 'red_i': 69, 'eco_i': 6, 'sunrpc': 43,
               'ftp_data': 14, 'urh_i': 62, 'pm_dump': 38, 'pop_3': 13,
               'pop_2': 52, 'systat': 30, 'ftp': 7, 'uucp': 37, 'whois': 21,
               'tftp_u': 66, 'netbios_dgm': 41, 'efs': 54, 'remote_job': 25,
               'sql_net': 57, 'daytime': 16, 'ntp_u': 8, 'finger': 4,
               'ldap': 42, 'netbios_ns': 60, 'kshell': 61, 'iso_tsap': 59,
               'ecr_i': 9, 'nntp': 36, 'http_2784': 63, 'shell': 33,
               'domain_u': 2, 'uucp_path': 56, 'courier': 44, 'exec': 45,
               'aol': 65, 'netstat': 15, 'telnet': 5, 'gopher': 24, 'rje': 26,
               'hostnames': 55, 'link': 29, 'ssh': 17, 'http_443': 48,
               'csnet_ns': 47, 'X11': 32, 'IRC': 39, 'harvest': 64,
               'imap4': 35, 'icmp': 70, 'supdup': 28, 'name': 20, 'nnsp': 53,
               'mtp': 23, 'http': 0, 'bgp': 46, 'ctf': 27, 'klogin': 49,
               'vmnet': 58, 'time': 19, 'discard': 31, 'login': 34, 'auth': 3,
               'other': 10, 'http_8001': 67}
flag_map = {'OTH': 4, 'RSTR': 8, 'S3': 3, 'S2': 1, 'S1': 2, 'S0': 7,
            'RSTOS0': 9, 'REJ': 5, 'SH': 10, 'RSTO': 6, 'SF': 0}
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
print(len(protocol_map), 'protocol types')
print(len(service_map), 'service types')
print(len(flag_map), 'flag types')
print(len(attack_map), 'attack types')
# print(len(protocol_map), 'protocol types: ', protocol_map)
# print(len(service_map), 'service types: ', service_map)
# print(len(flag_map), 'flag types: ', flag_map)
# print(len(attack_map), 'attack types: ', attack_map)


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
                       'sqlattack': 'other',
                       'worm': 'other'}
# If category_map[x] != 0, then x is a type of attack
category_map = {'normal': 0, 'probe': 1, 'dos': 2, 'u2r': 3, 'r2l': 4}
binary_map = {'normal': 0, 'probe': 1, 'dos': 1, 'u2r': 1, 'r2l': 1, 'other': 1}


def load_traffic(filename, traffic_map=category_map, show=6):
    """Each row  of all_traffic is a traffic record"""
    all_traffics = list()
    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        seen = set()
        for row in reader:
            try:
                # Ignore difficulty level and 19th feature,
                # which is a constant zero
                row = row[:-1]
                traffic = row[0:19] + row[20:]
                traffic[1] = protocol_map[row[1]]
                traffic[2] = service_map[row[2]]
                traffic[3] = flag_map[row[3]]
                attack = row[-1]

                category = attack_category_map[attack]
                traffic[-1] = traffic_map[category]
                traffic = [float(r) for r in traffic]
                all_traffics.append(traffic)
                if category not in seen and show > 0:
                    print(category, ' traffic')
                    show -= 1
                    seen.add(category)
            except KeyError as e:
                print('Cannot parse record %s:', e)

    return np.array(all_traffics)


def one_hot_encoding(labels):
    """Given labels which is a N dimentioanl vector,
    return a N by C matrix where each row is one-hot-encoding of each label"""
    encoding = np.zeros((labels.shape[0], num_classes), dtype=float)
    for [i, l] in enumerate(labels):
        encoding[i, int(l)] = 1.0
    return encoding


def mean_normalize(matrix):
    print(matrix[:, 19])
    print(matrix[:, 20])

    for i in range(matrix.shape[1]):
        mu = np.mean(matrix[:, i])
        sigma = np.std(matrix[:, i])
        matrix[:, i] = (matrix[:, i] - mu) / sigma
    return matrix


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
        # print('Mean normalize dataset...')
        # dataset = preprocessing.scale(dataset)
        # print(dataset.mean(axis=0))
        # print(dataset.std(axis=0))
        # print(dataset[np.random.randint(0, dataset.shape[0]), :])

        print('Convert label to one-hot-encoding...')
        labels = one_hot_encoding(labels)
        print(labels[:4, :])

        return dataset, labels
    else:
        # print('Mean normalize dataset...')
        # matrix = preprocessing.scale(matrix)
        # print(matrix.mean(axis=0))
        # print(matrix.std(axis=0))

        return matrix, None


def maybe_npsave(dataname, data, l, r, force=False):
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


def generate_train_valid_dataset(dataset, labels, percent=1.0, size=''):
    num_traffics = int(dataset.shape[0] * percent)
    left = 0
    right = int(0.9 * num_traffics)
    maybe_npsave('NSL-KDD/train_dataset' + size, dataset, left, right)
    maybe_npsave('NSL-KDD/train_ref' + size, labels, left, right)

    left = right
    right = num_traffics
    maybe_npsave('NSL-KDD/valid_dataset' + size, dataset, left, right)
    maybe_npsave('NSL-KDD/valid_ref' + size, labels, left, right)

    # train_dataset = dataset[:int(0.8 * num_traffics), :]
    # train_labels = labels[:int(0.8 * num_traffics), :]
    # valid_dataset = dataset[int(0.8 * num_traffics):, :]
    # valid_labels = labels[int(0.8 * num_traffics):, :]

    print('Training + Validation', dataset.shape, labels.shape)


def generate_test_dataset(dataset, labels, size=''):
    num_traffics = dataset.shape[0]
    print('Testing', dataset.shape, labels.shape)
    maybe_npsave('NSL-KDD/test_dataset' + size, dataset, 0, num_traffics)
    maybe_npsave('NSL-KDD/test_ref' + size, labels, 0, num_traffics)


def generate_datasets():
    global num_classes
    train = 'NSL-KDD/KDDTrain+.txt'
    if binary_label:
        print('Use binary label')
        num_classes = 2
        data_matrix = load_traffic(train, binary_map)
    else:
        num_classes = len(category_map)
        data_matrix = load_traffic(train)
    dataset, labels = shuffle_dataset_with_label(data_matrix)
    generate_train_valid_dataset(dataset, labels)

    test = 'NSL-KDD/KDDTest+.txt'
    if binary_label:
        print('Use binary label')
        num_classes = 2
        data_matrix = load_traffic(test, binary_map)
    else:
        num_classes = len(category_map)
        data_matrix = load_traffic(test)
    dataset, labels = shuffle_dataset_with_label(data_matrix)
    generate_test_dataset(dataset, labels)


if __name__ == '__main__':
    binary_label = False
    generate_datasets()
    binary_label = True
    generate_datasets()
