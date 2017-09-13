import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from netlearner.gan import GenerativeAdversarialNets, GANTwoLayers
from math import ceil
import subprocess

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_dataset = mnist.train.images
np.random.shuffle(train_dataset)
# normalizing input to [-1, 1] will not work

num_samples, input_dim = train_dataset.shape
noise_dim = 100
batch_size = 128
# num_epochs = 240
# init_lr = 0.0001
# num_steps = ceil(num_samples / batch_size * num_epochs)
# G_h = 360
# D_h = 360
# gan = GenerativeAdversarialNets(noise_dim, input_dim, # G_h, D_h)
num_epochs = 80
init_lr = 0.0001
num_steps = ceil(num_samples / batch_size * num_epochs)
G_hsize = [128, 128]
D_hsize = [128, 128]
gan = GANTwoLayers(noise_dim, input_dim,
                   G_hsize, D_hsize,
                   init_lr=init_lr, decay_steps=int(num_steps/10))
gan.train(batch_size, train_dataset, int(num_steps), init_lr)
gan.close()
subprocess.call("convert -delay 40 -dispose previous -loop 0 \
                %s/sample_*.png %s/animated.gif" %
                (gan.dirname, gan.dirname),
                shell=True)
