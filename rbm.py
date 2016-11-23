from __future__ import print_function
import numpy as np
from netlearner.utils import min_max_normalize, maybe_npsave
from netlearner.rbm import RestrictedBoltzmannMachine


raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
[train_dataset, valid_dataset, test_dataset] = min_max_normalize(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape)

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_hidden_rbm = 640
rbm_lr = 0.8
batch_size = 400
num_steps = 160000
rbm = RestrictedBoltzmannMachine(feature_size, num_hidden_rbm,
                                 batch_size, rbm_lr)
print('Restricted Boltzmann Machine built')
# rbm.train(train_dataset, batch_size, num_steps)
rbm.train_with_labels(train_dataset, train_labels, batch_size, num_steps)
test_loss = rbm.calc_reconstruct_loss(test_dataset)
print("Testset reconstruction error: %f" % test_loss)

hrand = np.random.random((train_dataset.shape[0], num_hidden_rbm))
encoded_train_dataset = rbm.encode_dataset(train_dataset, hrand)
hrand = np.random.random((test_dataset.shape[0], num_hidden_rbm))
encoded_test_dataset = rbm.encode_dataset(test_dataset, hrand)
print('Encoded training set', encoded_train_dataset.shape)
print('Encoded test set', encoded_test_dataset.shape)

tr_fn = maybe_npsave('encoded_trainset.rbm', encoded_train_dataset,
                     0, encoded_train_dataset.shape[0], True)
te_fn = maybe_npsave('encoded_testset.rbm', encoded_test_dataset,
                     0, encoded_test_dataset.shape[0], True)
print('Encoded train set', encoded_train_dataset.shape)
print('...saved to %s' % tr_fn)
print('Encoded test set', encoded_test_dataset.shape)
print('...saved to %s' % te_fn)
