import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from netlearner.gan import GenerativeAdversarialNets, GANTwoLayers
from math import ceil
import subprocess

np.random.seed(1654)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_dataset = mnist.train.images
np.random.shuffle(train_dataset)
num_samples, input_dim = train_dataset.shape

noise_dim = 100
batch_size = 128
keep_prob = 1.0


def create_gan():
    num_epochs = 180
    init_lr = 0.001
    num_steps = ceil(num_samples / batch_size * num_epochs)
    decay_steps = int(num_steps / 10)
    G_h = 256
    D_h = 256
    gan = GenerativeAdversarialNets(noise_dim, input_dim, G_h, D_h,
                                    init_lr, decay_steps)
    return gan, num_steps


def create_two_layer_gan():
    num_epochs = 200
    init_lr = 0.0005
    num_steps = ceil(num_samples / batch_size * num_epochs)
    decay_steps = int(num_steps / 20)
    G_hsize = [160, 128]
    D_hsize = [160, 128]
    gan = GANTwoLayers(noise_dim, input_dim,
                       G_hsize, D_hsize,
                       init_lr, decay_steps)
    return gan, num_steps


gan, num_steps = create_two_layer_gan()
gan.train(batch_size, train_dataset, int(num_steps), keep_prob)
gan.close()
subprocess.call("convert -delay 40 -dispose previous -loop 0 \
                %s/sample_*.png %s/animated.gif" %
                (gan.dirname, gan.dirname),
                shell=True)
