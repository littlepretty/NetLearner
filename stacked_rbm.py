from __future__ import print_function
import numpy as np
from netlearner.utils import min_max_normalize, accuracy, measure_prediction
from netlearner.stacked_rbm import StackedRBM


raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
test_labels = np.load('NSLKDD/test_ref.npy')

[train_dataset, valid_dataset, test_dataset] = min_max_normalize(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_hidden_sizes = [720, 360, 512, 1024]
num_labels = train_labels.shape[1]
srbm = StackedRBM(feature_size, num_hidden_sizes, num_labels)

batch_sizes = [800, 800, 800, 800]
num_steps = [160000, 160000, 160000, 160000]
ft_batch_size = 400
ft_num_steps = 160000
srbm.train(train_dataset, train_labels, batch_sizes, num_steps,
           ft_batch_size, ft_num_steps)
test_predict = srbm.make_prediction(test_dataset)
test_accuracy = accuracy(test_predict, test_labels)
measure_prediction(test_predict, test_labels, 'Test')
