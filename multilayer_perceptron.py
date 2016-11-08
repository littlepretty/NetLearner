from __future__ import print_function
import numpy as np
import tensorflow as tf
from netlearner.utils import accuracy, measure_prediction, standard_scale
from netlearner.multilayer_perceptron import MultilayerPerceptron


raw_train_dataset = np.load('NSL-KDD/train_dataset.npy')
train_labels = np.load('NSL-KDD/train_ref.npy')
raw_valid_dataset = np.load('NSL-KDD/valid_dataset.npy')
valid_labels = np.load('NSL-KDD/valid_ref.npy')
raw_test_dataset = np.load('NSL-KDD/test_dataset.npy')
test_labels = np.load('NSL-KDD/test_ref.npy')

# Mean normalize data
[train_dataset, valid_dataset, test_dataset] = standard_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
# merge train and valid dataset
train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# Run Classifier
# 1. Use Perceptron with 3 hidden layers
# 2. Train with very large batch size
num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]
hidden_layer_sizes = [400, 200, 100]
mp_classifier = MultilayerPerceptron(feature_size, hidden_layer_sizes,
                                     num_labels, trans_func=tf.nn.sigmoid,
                                     beta=0.0001, init_learning_rate=0.9)
batch_size = 10000
num_steps = 4000
mp_classifier.train(train_dataset, train_labels, batch_size, num_steps)

# Print out classification errors for all datasets
test_predict = mp_classifier.make_prediction(test_dataset)
test_accuracy = accuracy(test_predict, test_labels)
print("Testset total accuracy: %f" % test_accuracy)
measure_prediction(test_predict, test_labels, 'Test')
