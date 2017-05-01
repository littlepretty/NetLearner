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
hidden_layer_sizes = [40]
mp_classifier = MultilayerPerceptron(feature_size, hidden_layer_sizes,
                                     num_labels, beta=0.000,
                                     trans_func=tf.nn.relu,
                                     name='PureMLP')
batch_size = 100
num_epochs = 80
num_steps = ceil(train_dataset.shape[0] / batch_size * num_epochs)
init_lr = 0.1
mp_classifier.train_with_labels(train_dataset, train_labels,
                                batch_size, int(num_steps), init_lr,
                                valid_dataset, valid_labels,
                                test_dataset, test_labels, keep_prob=0.8)
f = open(mp_classifier.dirname + '/test.log')
print(f.read())
f.close()
weights = mp_classifier.get_weights('w0')
np.save(mp_classifier.dirname + '/w0.npy', weights)
