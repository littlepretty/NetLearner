from __future__ import print_function, division
import numpy as np
from netlearner.utils import min_max_scale, maybe_npsave
from netlearner.utils import hyperparameter_summary
from netlearner.rbm import RestrictedBoltzmannMachine
import tensorflow as tf
from math import ceil

tf.set_random_seed(9876)
encoder_name = 'RBM-UNSW2C'
raw_train_dataset = np.load('UNSW/train_dataset.npy')
train_labels = np.load('UNSW/train_labels.npy')
raw_valid_dataset = np.load('UNSW/valid_dataset.npy')
valid_labels = np.load('UNSW/valid_labels.npy')
raw_test_dataset = np.load('UNSW/test_dataset.npy')

[train_dataset, valid_dataset, test_dataset] = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape)

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_hidden_rbm = 400
rbm_lr = 0.01
batch_size = 10
num_epoch = 80
num_steps = ceil(train_dataset.shape[0] / batch_size * num_epoch)
rbm = RestrictedBoltzmannMachine(feature_size, num_hidden_rbm,
                                 batch_size, trans_func=tf.nn.sigmoid,
                                 num_labels=2,
                                 name=encoder_name)
rbm.train_with_labels(train_dataset, train_labels,
                      int(num_steps),
                      valid_dataset, rbm_lr)
test_loss = rbm.calc_reconstruct_loss(test_dataset)
print("Testset reconstruction error: %f" % test_loss)
hyperparameter = {'#hidden units': num_hidden_rbm,
                  'init_lr': rbm_lr,
                  'num_epochs': num_epoch,
                  'num_steps': num_steps,
                  'act_func': 'sigmoid',
                  'batch_size': batch_size, }
hyperparameter_summary(rbm.dirname, hyperparameter)

hrand = np.random.random((train_dataset.shape[0], num_hidden_rbm))
rbm_train_dataset = rbm.encode_dataset(train_dataset, hrand)
print('Encoded training set', rbm_train_dataset.shape)
hrand = np.random.random((valid_dataset.shape[0], num_hidden_rbm))
rbm_valid_dataset = rbm.encode_dataset(valid_dataset, hrand)
print('Encoded valid set', rbm_valid_dataset.shape)
hrand = np.random.random((test_dataset.shape[0], num_hidden_rbm))
rbm_test_dataset = rbm.encode_dataset(test_dataset, hrand)
print('Encoded test set', rbm_test_dataset.shape)

maybe_npsave('trainset.rbm', rbm_train_dataset, True)
maybe_npsave('validset.rbm', rbm_valid_dataset, True)
maybe_npsave('testset.rbm', rbm_test_dataset, True)
