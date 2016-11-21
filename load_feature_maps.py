from __future__ import print_function
import csv


def load_feature_maps(dataset='NSL-KDD',
                      files=['KDDTrain+.txt', 'KDDTest+.txt']):
    protocol_map = {}
    service_map = {}
    flag_map = {}
    land_map = {}
    login_map = {}
    host_login_map = {}
    guest_login_map = {}
    attack_map = {}
    for datafile in files:
        filename = dataset + '/' + datafile
        with open(filename, 'rb') as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for row in reader:
                if row[19] != '0':
                    print('row 19 is not useless')
                if row[1] not in protocol_map:
                    protocol_map[row[1]] = len(protocol_map)
                if row[2] not in service_map:
                    service_map[row[2]] = len(service_map)
                if row[3] not in flag_map:
                    flag_map[row[3]] = len(flag_map)
                if row[6] not in land_map:
                    land_map[row[6]] = len(land_map)
                if row[11] not in login_map:
                    login_map[row[11]] = len(login_map)
                if row[20] not in host_login_map:
                    host_login_map[row[20]] = len(host_login_map)
                if row[21] not in guest_login_map:
                    guest_login_map[row[21]] = len(guest_login_map)
                if dataset == 'KDDCup':
                    if row[-1] not in attack_map:
                        attack_map[row[-1]] = len(attack_map)
                else:
                    if row[-2] not in attack_map:
                        attack_map[row[-2]] = len(attack_map)

    print(len(protocol_map), 'protocol_types = \n', protocol_map)
    print(len(service_map), 'service_types = \n', service_map)
    print(len(flag_map), 'flag_types = \n', flag_map)
    print(len(land_map), 'land_types = \n', land_map)
    print(len(login_map), 'login_types = \n', login_map)
    print(len(host_login_map), 'host_login_types = \n', host_login_map)
    print(len(guest_login_map), 'guest_login_types = \n', guest_login_map)
    print(len(attack_map), 'attack_types = \n', attack_map)

    return [protocol_map, service_map, flag_map,
            land_map, login_map, host_login_map,
            guest_login_map, attack_map]

[protocol_map, service_map, flag_map,
 land_map, login_map, host_login_map,
 guest_login_map, attack_map] = load_feature_maps()

[protocol_map, service_map, flag_map,
 land_map, login_map, host_login_map,
 guest_login_map, attack_map] = load_feature_maps('KDDCup',
                                                  ['kddcup.traindata',
                                                   'kddcup.testdata'])
