from __future__ import print_function
import numpy as np
import tensorflow as tf
from netlearner.utils import hyperparameter_summary
from netlearner.utils import min_max_scale
from netlearner.multilayer_perceptron import MultilayerPerceptron
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
perm = np.random.permutation(test_dataset.shape[0])
test_dataset = test_dataset[perm, :]
test_labels = test_labels[perm, :]
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

num_samples, feature_size = train_dataset.shape
num_labels = train_labels.shape[1]
batch_size = 80
keep_prob = 0.80
beta = 0.0001
weights = [1.0, 100.0]
num_epochs = [10]
init_lrs = [0.001]
hidden_layer_sizes = [
                      [400],
                      # [800, 640], [160, 80], [80, 40],
                      # [400, 360, 320],
                      # [160, 120, 80], [120, 80, 40],
                      ]
for hidden_layer_size in hidden_layer_sizes:
    for init_lr in init_lrs:
        for num_epoch in num_epochs:
            num_steps = ceil(train_dataset.shape[0] / batch_size * num_epoch)
            mp_classifier = MultilayerPerceptron(feature_size,
                                                 hidden_layer_size,
                                                 num_labels, beta,
                                                 tf.nn.relu,
                                                 tf.nn.l2_loss, weights,
                                                 tf.train.AdamOptimizer,
                                                 name='PureMLP-UNSW2C')
            mp_classifier.train_with_labels(train_dataset, train_labels,
                                            batch_size, int(num_steps), init_lr,
                                            valid_dataset, valid_labels,
                                            test_dataset, test_labels,
                                            keep_prob)
            hyperparameter = {'hidden_layer_size': hidden_layer_size,
                              'init_lr': init_lr,
                              'num_epochs': num_epoch,
                              'num_steps': num_steps,
                              'regularization beta': beta,
                              'optimizer': 'AdamOptimizer',
                              'keep_prob': keep_prob,
                              'act_func': 'RELU',
                              'class_weights': weights,
                              'batch_size': batch_size, }
            hyperparameter_summary(mp_classifier.dirname,
                                   hyperparameter)
            f = open(mp_classifier.dirname + '/test.log')
            print(f.read())
            f.close()
            mp_classifier.exit()
