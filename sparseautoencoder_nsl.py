from __future__ import print_function, division
import numpy as np
from netlearner.utils import min_max_scale, maybe_npsave
from netlearner.autoencoder import SparseAutoencoder
import tensorflow as tf
from math import ceil

np.random.seed(4567)
tf.set_random_seed(4567)
encoder_name = 'SparseAE'
raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
train_dataset, valid_dataset, test_dataset = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_labels = np.load('NSLKDD/train_ref.npy')
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape)

feature_size = train_dataset.shape[1]
encoder_size = 64
init_lr = 0.01
autoencoder = SparseAutoencoder(feature_size, encoder_size,
                                optimizer=tf.train.AdamOptimizer,
                                sparsity=0.25, sparsity_weight=1,
                                name=encoder_name)
batch_size = 5000
num_epochs = 1000
num_steps = ceil(train_dataset.shape[0] / batch_size * num_epochs)
autoencoder.train_with_labels(train_dataset, train_labels,
                              batch_size, int(num_steps), init_lr,
                              valid_dataset)
test_loss = autoencoder.calc_reconstruct_loss(test_dataset)
print("Testset reconstruction loss: %f" % test_loss)

sae_train_dataset = autoencoder.encode_dataset(train_dataset)
sae_valid_dataset = autoencoder.encode_dataset(valid_dataset)
sae_test_dataset = autoencoder.encode_dataset(test_dataset)
tr_fn = maybe_npsave('trainset.' + encoder_name, sae_train_dataset,
                     0, sae_train_dataset.shape[0], True)
va_fn = maybe_npsave('validset.' + encoder_name, sae_valid_dataset,
                     0, sae_valid_dataset.shape[0], True)
te_fn = maybe_npsave('testset.' + encoder_name, sae_test_dataset,
                     0, sae_test_dataset.shape[0], True)
print('Encoded train set %s saved to %s' % (sae_train_dataset.shape, tr_fn))
print('Encoded valid set %s saved to %s' % (sae_valid_dataset.shape, va_fn))
print('Encoded test set %s saved to %s' % (sae_test_dataset.shape, te_fn))
