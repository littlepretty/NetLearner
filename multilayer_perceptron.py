from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from netlearner.utils import min_max_scale
from netlearner.multilayer_perceptron import MultilayerPerceptron
from math import ceil

np.random.seed(1234)
tf.set_random_seed(1234)
raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
test_labels = np.load('NSLKDD/test_ref.npy')
[train_dataset, valid_dataset, test_dataset] = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
perm = np.random.permutation(train_dataset.shape[0])
train_dataset = train_dataset[perm, :]
train_labels = train_labels[perm, :]
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]

batch_size = 100
num_epochs = 100
num_steps = ceil(train_dataset.shape[0] / batch_size * num_epochs)
init_lrs = [0.1, 0.05, 0.01]
hidden_layer_sizes = [[40], [80], [120], [160], [200],
                      [200, 160], [160, 120], [120, 80], [80, 40],
                      [200, 160, 120], [160, 120, 80], [120, 80, 40],
                      [200, 160, 120, 80], [160, 120, 80, 40],
                      [200, 160, 120, 80, 40]]
keep_probs = [0.8, 0.5, 0.2]
for hidden_layer_size in hidden_layer_sizes:
    for init_lr in init_lrs:
        for keep_prob in keep_probs:

            mp_classifier = MultilayerPerceptron(feature_size,
                                                 hidden_layer_size,
                                                 num_labels, beta=0.000,
                                                 trans_func=tf.nn.relu,
                                                 name='PureMLP')
            mp_classifier.train_with_labels(train_dataset, train_labels,
                                            batch_size, int(num_steps),
                                            init_lr, valid_dataset,
                                            valid_labels, test_dataset,
                                            test_labels, keep_prob=keep_prob)
            f = open(mp_classifier.dirname + '/test.log')
            print(f.read())
            f.close()
            weights = mp_classifier.get_weights('w0')
            np.save(mp_classifier.dirname + '/w0.npy', weights)
