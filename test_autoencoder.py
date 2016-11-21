from __future__ import print_function
import numpy as np
# from netlearner.autoencoder import MaskNoiseAutoencoder
from netlearner.autoencoder import SparseAutoencoder
# from netlearner.autoencoder import Autoencoder
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def test_mnist():
    mnist = input_data.read_data_sets(
        '~/Researches/MachineLearning/TFTutorial/MNIST_data',
        one_hot=True)
    train_dataset = mnist.train.images
    test_dataset = mnist.test.images
    print('Trainset', train_dataset.shape)
    print('Testset', test_dataset.shape)
    feature_size = train_dataset.shape[1]
    encoder_size = 1200
    # autoencoder = Autoencoder(
        # feature_size, encoder_size, encoder_lr=0.001,
        # l2_weight=0.003)
    # autoencoder = MaskNoiseAutoencoder(
        # feature_size, encoder_size, mask_prob=0.64, encoder_lr=0.001,
        # l2_weight=0.003)
    autoencoder = SparseAutoencoder(
        feature_size, encoder_size, sparsity=0.05,
        sparsity_weight=1, encode_lr=0.001, l2_weight=0.003)
    batch_size = 550
    num_steps = 1000
    autoencoder.train(train_dataset, batch_size, num_steps)
    test_loss = autoencoder.calc_reconstruct_loss(test_dataset)
    print("Testset decode loss: %f" % test_loss)

    rand_train = [train_dataset[np.random.randint(0, test_dataset.shape[0])]]
    reconstruct_train = autoencoder.reconstruct(rand_train)
    rand_train = np.multiply(255, np.reshape(rand_train, [28, 28]))
    reconstruct_train = np.multiply(255,
                                    np.reshape(reconstruct_train, [28, 28]))

    rand_test = [test_dataset[np.random.randint(0, test_dataset.shape[0])]]
    reconstruct_test = autoencoder.reconstruct(rand_test)
    rand_test = np.multiply(255, np.reshape(rand_test, [28, 28]))
    reconstruct_test = np.multiply(255,
                                   np.reshape(reconstruct_test, [28, 28]))

    plt.matshow(rand_test, cmap=plt.cm.gray)
    plt.title('Random test image')
    plt.figure(1)
    plt.matshow(reconstruct_test, cmap=plt.cm.gray)
    plt.title('Reconstructed test image')

    plt.figure(2)
    plt.matshow(rand_train, cmap=plt.cm.gray)
    plt.title('Random train image')
    plt.figure(3)
    plt.matshow(reconstruct_train, cmap=plt.cm.gray)
    plt.title('Reconstructed train image')
    plt.show()

test_mnist()
