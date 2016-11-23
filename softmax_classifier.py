from __future__ import print_function
import numpy as np
import tensorflow as tf
from netlearner.utils import accuracy, measure_prediction
from netlearner.multilayer_perceptron import MultilayerPerceptron

encoded_train_dataset = np.load('encoded_trainset.rbm.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
encoded_test_dataset = np.load('encoded_testset.rbm.npy')
test_labels = np.load('NSLKDD/test_ref.npy')

# use encoded traning and testing data
num_samples = encoded_train_dataset.shape[0]
feature_size = encoded_train_dataset.shape[1]
num_labels = train_labels.shape[1]
hidden_layer_sizes = [1000]
mp_classifier = MultilayerPerceptron(feature_size,
                                     hidden_layer_sizes, num_labels,
                                     trans_func=tf.nn.sigmoid,
                                     init_learning_rate=0.64,
                                     beta=0.0001)
batch_size = 240
num_steps = 1600000
mp_classifier.train_with_bias(encoded_train_dataset,
                              train_labels, batch_size, num_steps)
test_predict = mp_classifier.make_prediction(encoded_test_dataset)
test_accuracy = accuracy(test_predict, test_labels)
measure_prediction(test_predict, test_labels, 'Test')
