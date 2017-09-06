from __future__ import print_function
import numpy as np
import tensorflow as tf
from netlearner.utils import min_max_scale
from netlearner.gan import GenerativeAdversarialNets
from math import ceil

raw_train_dataset = np.load('UNSW/train_dataset.npy')
train_labels = np.load('UNSW/train_labels.npy')
raw_valid_dataset = np.load('UNSW/valid_dataset.npy')
valid_labels = np.load('UNSW/valid_labels.npy')
raw_test_dataset = np.load('UNSW/test_dataset.npy')
test_labels = np.load('UNSW/test_labels.npy')

[train_dataset, valid_dataset, test_dataset] = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
perm = np.random.permutation(train_dataset.shape[0])
train_dataset = train_dataset[perm, :]
train_labels = train_labels[perm, :]
print('Training set', train_dataset.shape, train_labels.shape)
# perm = np.random.permutation(test_dataset.shape[0])
# test_dataset = test_dataset[perm, :]
# test_labels = test_labels[perm, :]
# print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)

num_samples, input_dim = train_dataset.shape
num_labels = train_labels.shape[1]
noise_dim = 100
G_hidden_layer = 128
D_hidden_layer = 128

batch_size = 80
num_epochs = 4
init_lr = 0.001
num_steps = ceil(num_samples / batch_size * num_epochs)

gan = GenerativeAdversarialNets(noise_dim, input_dim,
                                G_hidden_layer, D_hidden_layer,
                                trans_func=tf.nn.relu)
gan.train(batch_size, train_dataset, int(num_steps), init_lr)
