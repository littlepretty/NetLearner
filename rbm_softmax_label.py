from __future__ import print_function
import numpy as np
from netlearner.utils import measure_prediction
from sklearn.preprocessing import MinMaxScaler
from netlearner.utils import attach_candidate_labels
from netlearner.rbm import RestrictedBoltzmannMachine
import tensorflow as tf


raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
test_labels = np.load('NSLKDD/test_ref.npy')

base_dataset = np.concatenate((raw_train_dataset, raw_valid_dataset), axis=0)
preprocessor = MinMaxScaler()
preprocessor.fit(base_dataset)
train_dataset = preprocessor.transform(raw_train_dataset)
valid_dataset = preprocessor.transform(raw_valid_dataset)
test_dataset = preprocessor.transform(raw_test_dataset)

train_dataset = np.concatenate((train_dataset, train_labels), axis=1)
valid_dataset = np.concatenate((valid_dataset, valid_labels), axis=1)
train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
# Build test dataset with 5 possible one-hot encoded label matrix
test_input = attach_candidate_labels(test_dataset, 5)

print('Training set', train_dataset.shape)
print('Test set', test_input.shape)

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_hidden_rbm = 32
rbm_lr = 0.1
batch_size = 10
num_steps = 800001
rbm = RestrictedBoltzmannMachine(feature_size, num_hidden_rbm,
                                 batch_size, rbm_lr,
                                 trans_func=tf.nn.sigmoid,
                                 name='LabeledSoftmaxRBM')
print('Restricted Boltzmann Machine built')
rbm.train_with_labels(train_dataset, train_labels, num_steps, valid_dataset)

energy = np.zeros([test_input.shape[1], 5])
for i in range(energy.shape[1]):
    energy[:, i] = rbm.calculate_free_energy(test_input[i]).flatten()

measure_prediction(np.multiply(energy, -1), test_labels)
