import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from netlearner.gan import GenerativeAdversarialNets
from math import ceil

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_dataset = mnist.train.images
# normalizing input to [-1, 1] will not work
train_labels = mnist.train.labels

num_samples, input_dim = train_dataset.shape
num_labels = train_labels.shape[1]
noise_dim = 100
G_hidden_layer = 100
D_hidden_layer = 100

batch_size = 128
num_epochs = 120
init_lr = 0.001
num_steps = ceil(num_samples / batch_size * num_epochs)
gan = GenerativeAdversarialNets(noise_dim, input_dim,
                                G_hidden_layer, D_hidden_layer,
                                trans_func=tf.nn.relu)
gan.train(batch_size, train_dataset, int(num_steps), init_lr)
gan.close()
