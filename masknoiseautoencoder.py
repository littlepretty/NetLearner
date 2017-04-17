from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from netlearner.utils import min_max_scale, maybe_npsave
from netlearner.autoencoder import MaskNoiseAutoencoder
from math import ceil

np.random.seed(4567)
tf.set_random_seed(4567)
encoder_name = 'MaskNoiseAE'
raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
train_dataset, valid_dataset, test_dataset = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape)

feature_size = train_dataset.shape[1]
encoder_size = 100
init_lr = 0.01
autoencoder = MaskNoiseAutoencoder(feature_size, encoder_size,
                                   mask_fraction=0.2)
batch_size = 10
num_epochs = 10
num_steps = ceil(train_dataset.shape[0] / batch_size * num_epochs)
autoencoder.train_with_labels(train_dataset, train_labels,
                              batch_size, int(num_steps), init_lr,
                              valid_dataset)
test_loss = autoencoder.calc_reconstruct_loss(test_dataset)
print("Testset decode loss: %f" % test_loss)

mae_train_dataset = autoencoder.encode_dataset(train_dataset)
mae_valid_dataset = autoencoder.encode_dataset(valid_dataset)
mae_test_dataset = autoencoder.encode_dataset(test_dataset)
tr_fn = maybe_npsave('trainset.' + encoder_name, mae_train_dataset,
                     0, mae_train_dataset.shape[0], True)
va_fn = maybe_npsave('validset.' + encoder_name, mae_valid_dataset,
                     0, mae_valid_dataset.shape[0], True)
te_fn = maybe_npsave('testset.' + encoder_name, mae_test_dataset,
                     0, mae_test_dataset.shape[0], True)
print('Encoded train set %s saved to %s' % (mae_train_dataset.shape, tr_fn))
print('Encoded valid set %s saved to %s' % (mae_valid_dataset.shape, va_fn))
print('Encoded test set %s saved to %s' % (mae_test_dataset.shape, te_fn))
