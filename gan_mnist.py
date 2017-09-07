import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from netlearner.gan import GenerativeAdversarialNets
from math import ceil

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_dataset = mnist.train.images
train_labels = mnist.train.labels

num_samples, input_dim = train_dataset.shape
num_labels = train_labels.shape[1]
noise_dim = 100
G_hidden_layer = 128
D_hidden_layer = 128

batch_size = 128
num_epochs = 200
init_lr = 0.001
num_steps = ceil(num_samples / batch_size * num_epochs)

gan = GenerativeAdversarialNets(noise_dim, input_dim,
                                G_hidden_layer, D_hidden_layer,
                                trans_func=tf.nn.relu)
gan.train(batch_size, train_dataset, int(num_steps), init_lr)
