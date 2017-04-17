from __future__ import print_function
import numpy as np
from netlearner.utils import min_max_scale, standard_scale, maybe_npsave
from netlearner.autoencoder import Autoencoder
import tensorflow as tf


np.random.seed(1234)
tf.set_random_seed(1234)
np.set_printoptions(precision=4)
raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
train_dataset, valid_dataset, test_dataset = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_labels = np.load('NSLKDD/train_ref.npy')
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape)

feature_size = train_dataset.shape[1]
encoder_size = 100
init_lr = 0.01
autoencoder = Autoencoder(feature_size, encoder_size,
                          transfer_func=tf.nn.sigmoid,
                          name='AE')
batch_size = 20
num_steps = 1001
autoencoder.train_with_labels(train_dataset, train_labels,
                              batch_size, num_steps, init_lr,
                              valid_dataset)
test_loss = autoencoder.calc_reconstruct_loss(test_dataset)
print("Testset reconstruction loss: %f" % test_loss)

ae_train_dataset = autoencoder.encode_dataset(train_dataset)
ae_valid_dataset = autoencoder.encode_dataset(valid_dataset)
ae_test_dataset = autoencoder.encode_dataset(test_dataset)
tr_fn = maybe_npsave('trainset.ae', ae_train_dataset,
                     0, ae_train_dataset.shape[0], True)
va_fn = maybe_npsave('validset.ae', ae_valid_dataset,
                     0, ae_valid_dataset.shape[0], True)
te_fn = maybe_npsave('testset.ae', ae_test_dataset,
                     0, ae_test_dataset.shape[0], True)
print('Encoded train set %s saved to %s' % (ae_train_dataset.shape, tr_fn))
print('Encoded valid set %s saved to %s' % (ae_valid_dataset.shape, va_fn))
print('Encoded test set %s saved to %s' % (ae_test_dataset.shape, te_fn))
