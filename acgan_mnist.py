import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from netlearner.ac_gan import AuxiliaryClassifierGAN, ACGANTwoLayers
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
keep_prob = 1.0


def create_ac_gan():
    print('Creating Single Layer AC-GAN')
    G_hidden_layer = 160
    D_hidden_layer = 160
    init_lr = 0.001
    num_epochs = 240
    num_steps = int(ceil(num_samples / batch_size) * num_epochs)
    decay_steps = num_steps / 10  # decay learning rate every 10 epochs
    gan = AuxiliaryClassifierGAN(noise_dim, input_dim, label_dim,
                                 G_hidden_layer, D_hidden_layer,
                                 init_lr, decay_steps)
    return gan, num_steps


def create_two_layer_ac_gan():
    print('Creating 2 Layer AC-GAN')
    G_hidden_layer = [200, 200]
    D_hidden_layer = [160, 80]
    init_lr = 0.0008
    num_epochs = 320
    num_steps = int(ceil(num_samples / batch_size) * num_epochs)
    decay_steps = num_steps / 8  # decay learning rate every 10 epochs
    gan = ACGANTwoLayers(noise_dim, input_dim, label_dim,
                         G_hidden_layer, D_hidden_layer,
                         init_lr, decay_steps)
    return gan, num_steps


gan, num_steps = create_two_layer_ac_gan()
gan.train(batch_size, train_dataset, train_labels,
          num_steps, keep_prob)
gan.close()
subprocess.call("convert -delay 40 -dispose previous -loop 0 \
                %s/sample_*.png %s/animated.gif" %
                (gan.dirname, gan.dirname),
                shell=True)
