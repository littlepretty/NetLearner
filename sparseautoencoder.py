from __future__ import print_function
import numpy as np
from netlearner.utils import min_max_normalize, maybe_npsave
from netlearner.autoencoder import SparseAutoencoder


raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
train_dataset, valid_dataset, test_dataset = min_max_normalize(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)

train_labels = np.load('NSLKDD/train_ref.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
train_labels = np.concatenate((train_labels, valid_labels), axis=0)

print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape)

feature_size = train_dataset.shape[1]
encoder_size = 500
encoder_lr = 0.0008
lambta = 0.003
beta = 1
autoencoder = SparseAutoencoder(feature_size, encoder_size,
                                encoder_lr, l2_weight=lambta,
                                sparsity_weight=beta)
batch_size = 300
num_steps = 100000
autoencoder.train_with_labels(train_dataset, train_labels,
                              batch_size, num_steps)
test_loss = autoencoder.calc_reconstruct_loss(test_dataset)
print("Testset reconstruction loss: %f" % test_loss)

encoded_train_dataset = autoencoder.encode_dataset(train_dataset)
encoded_test_dataset = autoencoder.encode_dataset(test_dataset)
tr_fn = maybe_npsave('encoded_trainset.sae', encoded_train_dataset,
                     0, encoded_train_dataset.shape[0], True)
te_fn = maybe_npsave('encoded_testset.sae', encoded_test_dataset,
                     0, encoded_test_dataset.shape[0], True)
print('Encoded train set', encoded_train_dataset.shape)
print('...saved to %s' % tr_fn)
print('Encoded test set', encoded_test_dataset.shape)
print('...saved to %s' % te_fn)
