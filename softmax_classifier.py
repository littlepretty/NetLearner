from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from netlearner.multilayer_perceptron import MultilayerPerceptron
from netlearner.utils import min_max_scale
from math import ceil

encoder_name = 'rbm'
np.random.seed(5678)
tf.set_random_seed(5678)
train_dataset = np.load('trainset.' + encoder_name + '.npy')
train_labels = np.load('UNSW/train_labels.npy')
valid_dataset = np.load('validset.' + encoder_name + '.npy')
valid_labels = np.load('UNSW/valid_labels.npy')
test_dataset = np.load('testset.' + encoder_name + '.npy')
test_labels = np.load('UNSW/test_labels.npy')

# raw_train_dataset = np.load('UNSW/train_dataset.npy')
# train_labels = np.load('UNSW/train_labels_bin.npy')
# raw_valid_dataset = np.load('UNSW/valid_dataset.npy')
# valid_labels = np.load('UNSW/valid_labels_bin.npy')
# raw_test_dataset = np.load('UNSW/test_dataset.npy')
# test_labels = np.load('UNSW/test_labels_bin.npy')

# [train_dataset, valid_dataset, test_dataset] = min_max_scale(
    # raw_train_dataset, raw_valid_dataset, raw_test_dataset)
# perm = np.random.permutation(train_dataset.shape[0])
# train_dataset = train_dataset[perm, :]
# train_labels = train_labels[perm, :]
# perm = np.random.permutation(test_dataset.shape[0])
# test_dataset = test_dataset[perm, :]
# test_labels = test_labels[perm, :]
# print('Training set', train_dataset.shape, train_labels.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)

num_samples, feature_size = train_dataset.shape
num_labels = train_labels.shape[1]
hidden_layer_sizes = []
weights = [1.0, 100.0]
mp = MultilayerPerceptron(feature_size, hidden_layer_sizes, num_labels,
                          trans_func=tf.nn.relu, beta=0.000,
                          optimizer=tf.train.AdamOptimizer,
                          class_weights=weights,
                          name='SoftmaxUse%s' % encoder_name.upper())
batch_size = 80
num_epochs = 80
num_steps = ceil(train_dataset.shape[0] / batch_size * num_epochs)
init_lr = 0.001
mp.train_with_labels(train_dataset, train_labels,
                     batch_size, int(num_steps), init_lr,
                     valid_dataset, valid_labels,
                     test_dataset, test_labels,
                     keep_prob=0.8)
f = open(mp.dirname + '/test.log')
print(f.read())
f.close()
