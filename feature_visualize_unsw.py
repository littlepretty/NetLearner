from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from netlearner.utils import min_max_scale, standard_scale


def load_feature_names():
    feature_str = "id,dur,spkts,dpkts,sbytes,dbytes,rate,sttl,\
        dttl,sload,dload,sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,\
        dwin,tcprtt,synack,ackdat,smean,dmean,trans_depth,response_body_len,\
        ct_srv_src,ct_state_ttl,ct_dst_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,\
        ct_dst_src_ltm,is_ftp_login,ct_ftp_cmd,ct_flw_http_mthd,ct_src_ltm,\
        ct_srv_dst,is_sm_ips_ports,proto,service,state,attack_cat,label"
    feature_names = feature_str.split(',')
    used_features = [name.strip() for name in feature_names[1: -2]]
    return used_features


def plot_dist_comp(data, label, name):
    """plot one column of dataset"""
    normal_indices = np.where(label == 1)[0]
    attack_indices = np.where(label == 0)[0]
    normal_traffic = data[normal_indices]
    attack_traffic = data[attack_indices]

    fig, ax = plt.subplots()
    n_nt, bins_nt, patches_nt = ax.hist(normal_traffic,
                                        bins='auto', normed=0, color='b',
                                        label='Normal', histtype='step')
    n_at, bins_at, patches_at = ax.hist(attack_traffic,
                                        bins='auto', normed=0, color='r',
                                        label='Attack', histtype='step')
    ax.set_xlabel('Feature %s values' % name)
    ax.set_ylabel('Probability density')
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig('UNSW/%s_dist_comp.png' % name, dpi=400)
    plt.close()


def plot_dist(data, name):
    """plot one column of dataset"""
    fig, ax = plt.subplots()
    n_nt, bins_nt, patches_nt = ax.hist(data, bins=100,
                                        normed=0, label=name)
    ax.set_xlabel('Feature %s values' % name)
    ax.set_ylabel('Probability density')
    ax.legend(loc='best')
    fig.tight_layout()
    plt.savefig('UNSW/%s_dist.png' % name, dpi=400)
    plt.show()
    plt.close()


def plot_feature_with_labels(dataset, labels):
    num_samples, feature_dim = dataset.shape
    print("%d features in total" % feature_dim)
    _, label_dim = labels.shape
    feature_names = load_feature_names()
    print("%d names" % len(feature_names))
    for i in range(1, 6):
        feature_name = feature_names[i]
        feature_data = dataset[:, i]
        feature_label = labels[:, 0]  # either 0 or 1 since one-hot encoding
        print('Range of %s: [%.4f, %.4f]' % (feature_name,
                                             np.min(feature_data),
                                             np.max(feature_data)))
        # delete = np.argmax(feature_data)
        # print('Deleting maximums at', delete)
        # feature_data = np.delete(feature_data, delete)
        # feature_label = np.delete(feature_label, delete)
        plot_dist(feature_data, feature_name)
        plot_dist_comp(feature_data, feature_label, feature_name)


raw_train_dataset = np.load('UNSW/train_dataset.npy')
train_labels = np.load('UNSW/train_labels.npy')
raw_valid_dataset = np.load('UNSW/valid_dataset.npy')
valid_labels = np.load('UNSW/valid_labels.npy')
raw_test_dataset = np.load('UNSW/test_dataset.npy')
test_labels = np.load('UNSW/test_labels.npy')

# [train_dataset, valid_dataset, test_dataset] = min_max_scale(
# raw_train_dataset, raw_valid_dataset, raw_test_dataset)
plot_feature_with_labels(raw_train_dataset, train_labels)
# [train_dataset, valid_dataset, test_dataset] = standard_scale(
# raw_train_dataset, raw_valid_dataset, raw_test_dataset)
# plot_normalized(train_dataset, train_labels)
