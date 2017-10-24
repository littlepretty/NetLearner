from __future__ import print_function
import csv
import io
import pandas
from sets import Set


def generate_header(feature_names):
    header = ''
    for name in feature_names:
        header += name + ','

    return header


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
        elif contents[2] in ['integer', 'timestamp']:
            discrete.append(contents[1])

    return feature_names, symbolic, continuous, discrete


# Output of discover_category_map
symbolic_features = {
    'srcip': ['10.40.170.2', '10.40.182.6', '149.171.126.19', '10.40.85.30',
              '149.171.126.18', '127.0.0.1', '10.40.85.10', '10.40.182.1',
              '175.45.176.2', '175.45.176.3', '175.45.176.0', '175.45.176.1',
              '149.171.126.7', '149.171.126.6', '149.171.126.5',
              '149.171.126.4', '149.171.126.3', '149.171.126.2',
              '149.171.126.1', '149.171.126.0', '149.171.126.17', '10.40.85.1',
              '149.171.126.9', '149.171.126.16',
              '149.171.126.8', '149.171.126.15', '149.171.126.11',
              '192.168.241.243', '149.171.126.14', '59.166.0.3',
              '149.171.126.12', '59.166.0.7', '59.166.0.6',
              '59.166.0.5', '59.166.0.4', '10.40.182.3', '59.166.0.2',
              '59.166.0.1', '59.166.0.0', '149.171.126.10',
              '59.166.0.9', '59.166.0.8', '149.171.126.13'],
    'dstip': ['10.40.182.255', '10.40.170.2', '10.40.182.6', '59.166.0.1',
              '59.166.0.9', '127.0.0.1', '10.40.85.30', '59.166.0.8',
              '224.0.0.1', '224.0.0.5', '10.40.198.10', '175.45.176.2',
              '175.45.176.3', '149.171.126.9', '149.171.126.8',
              '149.171.126.7', '149.171.126.6', '149.171.126.5',
              '149.171.126.4', '149.171.126.3', '149.171.126.2',
              '149.171.126.1', '149.171.126.0', '59.166.0.7', '10.40.85.1',
              '175.45.176.0', '59.166.0.6', '175.45.176.1', '59.166.0.5',
              '192.168.241.243', '59.166.0.4', '149.171.126.13',
              '192.168.241.50', '59.166.0.2', '149.171.126.17',
              '149.171.126.16', '149.171.126.15', '149.171.126.14',
              '10.40.182.3', '149.171.126.12', '149.171.126.11',
              '149.171.126.10', '59.166.0.0', '149.171.126.19',
              '149.171.126.18', '59.166.0.3', '32.50.32.66'],
    'proto': ['skip', 'ttp', 'tcp', 'chaos', 'rtp', 'netblt', 'tcf', 'crtp',
              'ax.25', 'ptp', 'merit-inp', 'xtp', 'crudp', 'ipcomp', 'a/n',
              'aris', 'bna', 'rsvp', 'nvp', 'isis', 'iatp', '3pc', 'iso-ip',
              'pri-enc', 'bbn-rcc', 'emcon', 'wsn', 'idpr', 'br-sat-mon',
              'cftp', 'arp', 'pvp', 'sep', 'ipip', 'iplt', 'leaf-1',
              'pnni', 'zero', 'leaf-2', 'larp', 'esp', 'ddp', 'mux', 'smp',
              'xns-idp', 'vrrp', 'sctp', 'idpr-cmtp', 'ipx-n-ip', 'dcn',
              'ipv6-route', 'ggp', 'qnx', 'ddx', 'xnet', 'sccopmce', 'pup',
              'tp++', 'rdp', 'mfe-nsp', 'argus', 'srp',
              'fc', 'idrp', 'ospf', 'vmtp', 'nsfnet-igp', 'pgm',
              'wb-expak', 'ippc', 'tlsp', 'igmp', 'unas', 'sun-nd', 'swipe',
              'aes-sp3-d', 'iso-tp4', 'st2', 'ipv6',
              'cphb', 'compaq-peer', 'udp', 'sps', 'l2tp', 'udt', 'fire',
              'mhrp', 'hmp', 'micp', 'ipv6-opts', 'icmp', 'trunk-2',
              'trunk-1', 'mtp', 'uti', 'secure-vmtp', 'eigrp', 'cpnx',
              'stp', 'sdrp', 'ipv6-no', 'pim', 'rvd',
              'prm', 'ip', 'sprite-rpc', 'il', 'cbt', 'i-nlsp', 'ifmp',
              'ib', 'any', 'sat-mon', 'igp', 'narp', 'gre', 'encap', 'irtp',
              'sat-expak', 'gmtp', 'wb-mon', 'dgp', 'visa', 'etherip',
              'snp', 'ipnip', 'mobile', 'scps',
              'egp', 'kryptolan', 'vines', 'pipe', 'ipv6-frag', 'ipcv', 'sm'],
    'state': ['ACC', 'PAR', 'CLO', 'MAS', 'no', 'INT', 'URN', 'REQ', 'URH',
              'TXD', 'ECO', 'ECR', 'RST', 'TST', 'FIN', 'CON'],
    'service': ['ftp', 'http', 'snmp', '-', 'smtp', 'pop3', 'ssl',
                'ftp-data', 'radius', 'ssh', 'dns', 'dhcp', 'irc']
}


def discovery_category_map(filenames):
    srcip, dstip, proto = Set(), Set(), Set()
    state, service = Set(), Set()
    for filename in filenames:
        csv_file = io.open(filename, 'r', encoding='utf-8-sig')
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            srcip.add(row[0])
            dstip.add(row[2])
            proto.add(row[4])
            state.add(row[5])
            service.add(row[13])

    srcip = list(srcip)
    dstip = list(dstip)
    proto = list(proto)
    state = list(state)
    service = list(service)
    result = {'srcip': srcip, 'dstip': dstip, 'proto': proto,
              'state': state, 'service': service}
    for (key, value) in result:
        print(key, value)

    return result


def discovery_discrete_range(filenames, dnames, headers):
    max_list = {x: 0 for x in dnames}
    min_list = {x: 2424250010 for x in dnames}  # stime starts at 1.4 billion
    for filename in filenames:
        data_frame = pandas.read_csv(filename,
                                     sep=',',
                                     names=headers,
                                     engine='python',
                                     na_values='-',
                                     nrows=1000)
        # print(data_frame.shape)
        for name in dnames:
            column = data_frame[name]
            max_list[name] = max(max_list[name], column.max(skipna=True))
            min_list[name] = min(min_list[name], column.min(skipna=True))

    for name in dnames:
        print('%s ranges: [%s, %s]' % (name, min_list[name], max_list[name]))

    return max_list, min_list


if __name__ == '__main__':
    filenames = ['UNSW/UNSW-NB15_%d.csv' % x for x in range(1, 5)]
    # discovery_category_map(filenames)
    filename = 'UNSW/feature_names.csv'
    headers, _, _, dnames = get_feature_names(filename)
    discovery_discrete_range(filenames, dnames, headers)
