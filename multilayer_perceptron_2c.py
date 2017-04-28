from __future__ import print_function
import numpy as np
import tensorflow as tf
from netlearner.utils import accuracy, measure_prediction
from netlearner.utils import min_max_normalize
from netlearner.multilayer_perceptron import MultilayerPerceptron


raw_train_dataset = np.load('NSLKDD/train_dataset_bin.npy')
train_labels = np.load('NSLKDD/train_ref_bin.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset_bin.npy')
valid_labels = np.load('NSLKDD/valid_ref_bin.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset_bin.npy')
test_labels = np.load('NSLKDD/test_ref_bin.npy')

[train_dataset, valid_dataset, test_dataset] = min_max_normalize(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
perm = np.random.permutation(train_dataset.shape[0])
train_dataset = train_dataset[perm, :]
train_labels = train_labels[perm, :]
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

num_samples, feature_size = train_dataset.shape
num_labels = train_labels.shape[1]
hidden_layer_sizes = [64, 51]
mp_classifier = MultilayerPerceptron(feature_size, hidden_layer_sizes,
                                     num_labels, beta=0.000,
                                     trans_func=tf.nn.relu,
                                     name='PureMLP2C')
batch_size = 100
num_steps = 80000
init_lr = 0.4
mp_classifier.train_with_labels(
    train_dataset, train_labels, batch_size, num_steps, init_lr,
    valid_dataset, valid_labels, keep_prob=0.6)
test_predict = mp_classifier.make_prediction(test_dataset)
test_accuracy = accuracy(test_predict, test_labels)
measure_prediction(test_predict, test_labels, 'Test')
