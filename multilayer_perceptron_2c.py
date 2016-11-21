from __future__ import print_function
import numpy as np
from netlearner.utils import accuracy, measure_prediction
from netlearner.utils import min_max_normalize
from netlearner.multilayer_perceptron import MultilayerPerceptron


raw_train_dataset = np.load('NSLKDD/train_dataset_bin.npy')
train_labels = np.load('NSLKDD/train_ref_bin.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset_bin.npy')
valid_labels = np.load('NSLKDD/valid_ref_bin.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset_bin.npy')
test_labels = np.load('NSLKDD/test_ref_bin.npy')

# Mean normalize data
[train_dataset, valid_dataset, test_dataset] = min_max_normalize(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
# merge train and valid dataset
train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]
hidden_layer_sizes = [512, 400, 300, 256, 96]

# feature_columns = learn.infer_real_valued_columns_from_input(train_dataset)
# dnn = learn.DNNClassifier(hidden_units=hidden_layer_sizes,
# feature_columns=feature_columns,
# n_classes=5, dropout=0.8)
# y_train = np.argmax(train_labels, 1)
# dnn.fit(train_dataset, y_train, steps=100)

# test_predict = dnn.predict(test_dataset)
# y_test = np.argmax(test_labels, 1)
# score = metrics.accuracy_score(y_test, test_predict)
# print('Accuracy:', score)
# class_table = np.zeros((num_labels, num_labels))
# for (a, p) in zip(y_test, test_predict):
# class_table[a][p] += 1

# headers = [str(i) for i in range(num_labels)]
# print(tabulate(class_table, headers))

mp_classifier = MultilayerPerceptron(feature_size, hidden_layer_sizes,
                                     num_labels, beta=0.0001,
                                     init_learning_rate=0.9)
batch_size = 1000
num_steps = 80000
mp_classifier.train_with_bias(
    train_dataset, train_labels, batch_size, num_steps, keep_prob=0.9)
test_predict = mp_classifier.make_prediction(test_dataset)
test_accuracy = accuracy(test_predict, test_labels)
measure_prediction(test_predict, test_labels, 'Test')
