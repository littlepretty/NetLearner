from __future__ import print_function
import numpy as np
from netlearner.utils import accuracy, measure_prediction, min_max_normalize
from netlearner.autoencoder import MaskingNoiseAutoencoder
from netlearner.multilayer_perceptron import MultilayerPerceptron


raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
test_labels = np.load('NSLKDD/test_ref.npy')
train_dataset, valid_dataset, test_dataset = min_max_normalize(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
# merge training and validation data
train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

feature_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]
encoder_size = 1000
encoder_lr = 0.1
beta = 0.01
mask_prob = 0.64
autoencoder = MaskingNoiseAutoencoder(
    feature_size, encoder_size, mask_prob, encoder_lr, beta)
batch_size = 2000
num_steps = 80000
autoencoder.train(train_dataset, batch_size, num_steps)
test_loss = autoencoder.calc_reconstruct_loss(test_dataset)
print("Testset decode loss: %f" % test_loss)
encoded_train_dataset = autoencoder.encode_dataset(train_dataset)
encoded_test_dataset = autoencoder.encode_dataset(test_dataset)
print('Encoded train set', encoded_train_dataset.shape, train_labels.shape)
print('Encoded test set', encoded_test_dataset.shape, test_labels.shape)
# use encoded traning and testing data
num_samples = encoded_train_dataset.shape[0]
feature_size = encoded_train_dataset.shape[1]
num_labels = train_labels.shape[1]
hidden_layer_sizes = []
mp_classifier = MultilayerPerceptron(feature_size,
                                     hidden_layer_sizes, num_labels)
batch_size = 240
num_steps = 120000
mp_classifier.train(encoded_train_dataset, train_labels, batch_size, num_steps)
test_predict = mp_classifier.make_prediction(encoded_test_dataset)
test_accuracy = accuracy(test_predict, test_labels)
measure_prediction(test_predict, test_labels, 'Test')
