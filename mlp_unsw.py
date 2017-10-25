from __future__ import print_function
import numpy as np
import tensorflow as tf
from preprocess.unsw import generate_dataset
from netlearner.utils import hyperparameter_summary
from netlearner.utils import augment_quantiled, permutate_dataset
from netlearner.multilayer_perceptron import MultilayerPerceptron

generate_dataset(True)
raw_train_dataset = np.load('UNSW/train_dataset.npy')
train_labels = np.load('UNSW/train_labels.npy')
raw_valid_dataset = np.load('UNSW/valid_dataset.npy')
valid_labels = np.load('UNSW/valid_labels.npy')
raw_test_dataset = np.load('UNSW/test_dataset.npy')
test_labels = np.load('UNSW/test_labels.npy')

columns = np.array(range(1, 6) + range(8, 16) + range(17, 19) +
                   range(23, 25) + [26])
[train_dataset, valid_dataset, test_dataset] = augment_quantiled(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset, columns)
permutate_dataset(train_dataset, train_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

num_samples, feature_size = train_dataset.shape
num_labels = train_labels.shape[1]
batch_size = 80
keep_prob = 0.80
beta = 0.00008
weights = [1.0, 1.0]
num_epochs = [160]
init_lrs = [0.001]
hidden_layer_sizes = [
                      [400, 400, 400, 400],
                      # [800, 640], [160, 80], [80, 40],
                      # [400, 360, 320],
                      # [160, 120, 80], [120, 80, 40],
                      ]
for hidden_layer_size in hidden_layer_sizes:
    for init_lr in init_lrs:
        for num_epoch in num_epochs:
            num_steps = int(train_dataset.shape[0] / batch_size * num_epoch)
            decay_steps = num_steps // num_epoch
            mp_classifier = MultilayerPerceptron(feature_size,
                                                 hidden_layer_size,
                                                 num_labels, init_lr,
                                                 decay_steps, beta,
                                                 tf.nn.relu,
                                                 tf.nn.l2_loss, weights,
                                                 tf.train.AdamOptimizer,
                                                 name='PureMLP-UNSW2C')
            mp_classifier.train_with_labels(train_dataset, train_labels,
                                            batch_size, num_steps,
                                            valid_dataset, valid_labels,
                                            test_dataset, test_labels,
                                            keep_prob)
            hyperparameter = {'hidden_layer_size': hidden_layer_size,
                              'init_lr': init_lr,
                              'num_epochs': num_epoch,
                              'num_steps': num_steps,
                              'regularization beta': beta,
                              'optimizer': 'AdamOptimizer',
                              'keep_prob': keep_prob,
                              'act_func': 'RELU',
                              'class_weights': weights,
                              'batch_size': batch_size, }
            hyperparameter_summary(mp_classifier.dirname,
                                   hyperparameter)
            f = open(mp_classifier.dirname + '/test.log')
            print(f.read())
            f.close()
            mp_classifier.exit()
