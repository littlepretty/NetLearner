import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from netlearner.ac_gan import AuxiliaryClassifierGAN
from math import ceil
import subprocess

np.random.seed(7453)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_dataset = mnist.train.images
train_labels = mnist.train.labels
num_samples, input_dim = train_dataset.shape
_, label_dim = train_labels.shape

perm = np.random.permutation(num_samples)
train_dataset = train_dataset[perm, :]
train_labels = train_labels[perm, :]

noise_dim = 100
batch_size = 128
num_epochs = 160
init_lr = 0.001
num_steps = int(ceil(num_samples / batch_size) * num_epochs)
decay_steps = num_steps / 10  # decay learning rate every 10 epochs
G_hidden_layer = 256
D_hidden_layer = 256
gan = AuxiliaryClassifierGAN(noise_dim, input_dim, label_dim,
                             G_hidden_layer, D_hidden_layer,
                             init_lr, decay_steps)
gan.train(batch_size, train_dataset, train_labels, num_steps)
gan.close()
subprocess.call("convert -delay 40 -dispose previous -loop 0 \
                %s/sample_*.png %s/animated.gif" %
                (gan.dirname, gan.dirname),
                shell=True)
