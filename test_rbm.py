from __future__ import print_function
import numpy as np
from netlearner.rbm import RestrictedBoltzmannMachine
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def test_mnist_with_rbm():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    train_dataset = mnist.train.images
    test_dataset = mnist.test.images
    print('Trainset', train_dataset.shape)
    print('Testset', test_dataset.shape)
    feature_size = train_dataset.shape[1]
    encoder_size = 1000
    batch_size = 550
    num_steps = 1000
    rbm = RestrictedBoltzmannMachine(
        feature_size, encoder_size, batch_size, lr=0.1)
    rbm.train(train_dataset, batch_size, num_steps)
    rbm.test_reconstruction(test_dataset)

    rand_train = [train_dataset[np.random.randint(0, test_dataset.shape[0])]]
    vrand = np.random.random((1, feature_size))
    hrand = np.random.random((1, encoder_size))
    reconstruct_train = rbm.reconstruct_dataset(rand_train, vrand, hrand)
    print(reconstruct_train.shape)
    rand_train = np.multiply(255, np.reshape(rand_train, [28, 28]))
    reconstruct_train = np.multiply(255,
                                    np.reshape(reconstruct_train, [28, 28]))

    rand_test = [test_dataset[np.random.randint(0, test_dataset.shape[0])]]
    vrand = np.random.random((1, feature_size))
    hrand = np.random.random((1, encoder_size))
    reconstruct_test = rbm.reconstruct_dataset(rand_test, vrand, hrand)
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

test_mnist_with_rbm()
