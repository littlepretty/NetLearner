from __future__ import print_function
import numpy as np
import tensorflow as tf
from netlearner.utils import accuracy, measure_prediction, standard_scale
from netlearner.autoencoder import SparseAutoencoder
from netlearner.multilayer_perceptron import MultilayerPerceptron


raw_train_dataset = np.load('NSL-KDD/train_dataset.npy')
train_labels = np.load('NSL-KDD/train_ref.npy')
raw_valid_dataset = np.load('NSL-KDD/valid_dataset.npy')
valid_labels = np.load('NSL-KDD/valid_ref.npy')
raw_test_dataset = np.load('NSL-KDD/test_dataset.npy')
test_labels = np.load('NSL-KDD/test_ref.npy')
train_dataset, valid_dataset, test_dataset = standard_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
print(np.min(train_dataset, axis=0))
print(np.max(train_dataset, axis=0))
# merge training and validation data
train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

feature_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]
encoder_size = 200
encoder_lr = 0.1
beta = 0.001
autoencoder = SparseAutoencoder(feature_size,
                                encoder_size, encoder_lr, beta=beta)
batch_size = 8000
num_steps = 1000
autoencoder.train(train_dataset, batch_size, num_steps)
test_loss = autoencoder.calc_total_loss(test_dataset)
print("Testset decode loss: %f" % test_loss)
encoded_train_dataset = autoencoder.encode_dataset(train_dataset)
encoded_valid_dataset = autoencoder.encode_dataset(valid_dataset)
encoded_test_dataset = autoencoder.encode_dataset(test_dataset)
print('Encoded train set', encoded_train_dataset.shape, train_labels.shape)
print('Encoded valid set', encoded_valid_dataset.shape, valid_labels.shape)
print('Encoded test set', encoded_test_dataset.shape, test_labels.shape)
# use encoded traning and testing data
num_samples = encoded_train_dataset.shape[0]
feature_size = encoded_train_dataset.shape[1]
num_labels = train_labels.shape[1]
hidden_layer_sizes = [128]
mp_classifier = MultilayerPerceptron(feature_size,
                                     hidden_layer_sizes, num_labels,
                                     trans_func=tf.nn.sigmoid,
                                     init_learning_rate=0.6, beta=0.0001)
batch_size = 100000
num_steps = 5000
mp_classifier.train(encoded_train_dataset, train_labels, batch_size, num_steps)
test_predict = mp_classifier.make_prediction(encoded_test_dataset)
test_accuracy = accuracy(test_predict, test_labels)
print("Testset total accuracy: %f" % test_accuracy)
measure_prediction(test_predict, test_labels, 'Test')
