from __future__ import print_function, division
import numpy as np
from netlearner.utils import min_max_scale, maybe_npsave
from netlearner.rbm import RestrictedBoltzmannMachine
import tensorflow as tf
from math import ceil

tf.set_random_seed(9876)
encoder_name = 'RBM'
raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
[train_dataset, valid_dataset, test_dataset] = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape)

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_hidden_rbm = 100
rbm_lr = 0.01
batch_size = 10
num_epochs = 40
num_steps = ceil(train_dataset.shape[0] / batch_size * num_epochs)
rbm = RestrictedBoltzmannMachine(feature_size, num_hidden_rbm,
                                 batch_size, trans_func=tf.nn.sigmoid,
                                 name=encoder_name)
print('Restricted Boltzmann Machine built')
rbm.train_with_labels(train_dataset, train_labels, int(num_steps),
                      valid_dataset, rbm_lr)
test_loss = rbm.calc_reconstruct_loss(test_dataset)
print("Testset reconstruction error: %f" % test_loss)

hrand = np.random.random((train_dataset.shape[0], num_hidden_rbm))
rbm_train_dataset = rbm.encode_dataset(train_dataset, hrand)
hrand = np.random.random((valid_dataset.shape[0], num_hidden_rbm))
rbm_valid_dataset = rbm.encode_dataset(valid_dataset, hrand)
hrand = np.random.random((test_dataset.shape[0], num_hidden_rbm))
rbm_test_dataset = rbm.encode_dataset(test_dataset, hrand)
print('Encoded training set', rbm_train_dataset.shape)
print('Encoded valid set', rbm_valid_dataset.shape)
print('Encoded test set', rbm_test_dataset.shape)
tr_fn = maybe_npsave('trainset.' + encoder_name, rbm_train_dataset,
                     0, rbm_train_dataset.shape[0], True)
va_fn = maybe_npsave('validset.' + encoder_name, rbm_valid_dataset,
                     0, rbm_valid_dataset.shape[0], True)
te_fn = maybe_npsave('testset.' + encoder_name, rbm_test_dataset,
                     0, rbm_test_dataset.shape[0], True)
print('Encoded train set %s saved to %s' % (rbm_train_dataset.shape, tr_fn))
print('Encoded valid set %s saved to %s' % (rbm_valid_dataset.shape, va_fn))
print('Encoded test set %s saved to %s' % (rbm_test_dataset.shape, te_fn))
