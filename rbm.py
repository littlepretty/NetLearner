from __future__ import print_function
import numpy as np
from numpy.random import binomial
import tensorflow as tf
from netlearner.utils import min_max_normalize, accuracy, measure_prediction
from netlearner.rbm import RestrictedBoltzmannMachine
from netlearner.multilayer_perceptron import MultilayerPerceptron


raw_train_dataset = np.load('NSL-KDD/train_dataset.npy')
train_labels = np.load('NSL-KDD/train_ref.npy')
raw_valid_dataset = np.load('NSL-KDD/valid_dataset.npy')
valid_labels = np.load('NSL-KDD/valid_ref.npy')
raw_test_dataset = np.load('NSL-KDD/test_dataset.npy')
test_labels = np.load('NSL-KDD/test_ref.npy')
[train_dataset, valid_dataset, test_dataset] = min_max_normalize(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]
num_hidden_rbm = 256
rbm_lr = 0.01
batch_size = 1000
num_steps = 4000
rbm = RestrictedBoltzmannMachine(feature_size, num_hidden_rbm,
                                 batch_size, rbm_lr)
print('Restricted Boltzmann Machine built')
rbm.train(train_dataset, batch_size, num_steps)
rbm.test_reconstruction(test_dataset)
# Encode datasets
hrand = binomial(1, 0.5, size=(train_dataset.shape[0], num_hidden_rbm))
encoded_train_dataset = rbm.encode_dataset(train_dataset, hrand)

hrand = binomial(1, 0.5, size=(valid_dataset.shape[0], num_hidden_rbm))
encoded_valid_dataset = rbm.encode_dataset(valid_dataset, hrand)

hrand = binomial(1, 0.5, size=(test_dataset.shape[0], num_hidden_rbm))
encoded_test_dataset = rbm.encode_dataset(test_dataset, hrand)
print('Encoded training set', encoded_train_dataset.shape, train_labels.shape)
print('Encoded validation set', encoded_valid_dataset.shape, valid_labels.shape)
print('Encoded test set', encoded_test_dataset.shape, test_labels.shape)
# Apppend a Multilayer Perceptron with 1 hidden layer
feature_size = encoded_train_dataset.shape[1]
num_labels = train_labels.shape[1]
hidden_layer_sizes = [400]
mp_classifier = MultilayerPerceptron(feature_size,
                                     hidden_layer_sizes, num_labels,
                                     trans_func=tf.nn.sigmoid, beta=0.0001)
batch_size = 1000
num_steps = 8000
mp_classifier.train(encoded_train_dataset, train_labels, batch_size, num_steps)
test_predict = mp_classifier.make_prediction(encoded_test_dataset)
test_accuracy = accuracy(test_predict, test_labels)
print("Testset total accuracy: %f" % test_accuracy)
measure_prediction(test_predict, test_labels, 'Test')
