from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from netlearner.multilayer_perceptron import MultilayerPerceptron
from math import ceil

encoder_name = 'RBM'
np.random.seed(5678)
tf.set_random_seed(5678)
train_dataset = np.load('trainset.' + encoder_name + '.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
valid_dataset = np.load('validset.' + encoder_name + '.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
test_dataset = np.load('testset.' + encoder_name + '.npy')
test_labels = np.load('NSLKDD/test_ref.npy')

num_samples, feature_size = train_dataset.shape
num_labels = train_labels.shape[1]
hidden_layer_sizes = []
mp = MultilayerPerceptron(feature_size, hidden_layer_sizes, num_labels,
                          trans_func=tf.nn.sigmoid,
                          optimizer=tf.train.AdamOptimizer,
                          beta=0.000, name='MLPUse%s' % encoder_name)
batch_size = 5000
num_epochs = 100
num_steps = ceil(train_dataset.shape[0] / batch_size * num_epochs)
init_lr = 0.01
mp.train_with_labels(train_dataset, train_labels,
                     batch_size, int(num_steps), init_lr,
                     valid_dataset, valid_labels,
                     test_dataset, test_labels,
                     keep_prob=0.8)
f = open(mp.dirname + '/test.log')
print(f.read())
f.close()
