import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from netlearner.utils import min_max_normalize
from netlearner.autoencoder import SparseAutoencoder
from netlearner.rbm import RestrictedBoltzmannMachine


def plot_2d_scatter(X, y, t):
    global plot_cnt
    fig = plt.figure(plot_cnt)
    plot_cnt += 1
    ax = fig.add_subplot(111)
    for (i, name) in enumerate(label_names):
        print('class ', name, ' amount = ', X[y == i, :].shape)
        ax.scatter(X[y == i, 0], X[y == i, 1],
                   c=colors[i], marker=markers[i],
                   lw=0.1, label=name)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)
    ax.set_title(t)


def reconstruct_error(X1, X2):
    error = np.sum(np.sum(
        np.square(np.subtract(X1, X2)))) / (X1.shape[0] * X1.shape[1])
    return 0.5 * error


np.random.seed(5)


raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
test_labels = np.load('NSLKDD/test_ref.npy')
train_dataset, valid_dataset, test_dataset = min_max_normalize(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

markers = ['d', 'v', '*', '^', 'o']
colors = ['c', 'y', 'g', 'b', 'r']
# label_names = ['normal', 'attack']
label_names = ['normal', 'probe', 'dos', 'u2r', 'r2l']
plot_cnt = 1

trY = np.argmax(train_labels, 1)
teY = np.argmax(test_labels, 1)


def pca_dataset(train_dataset, test_dataset, nc):
    pca = decomposition.PCA(n_components=nc)
    pca.fit(train_dataset)

    trX = pca.transform(train_dataset)
    recover_trX = pca.inverse_transform(trX)

    teX = pca.transform(test_dataset)
    recover_teX = pca.inverse_transform(teX)

    return [trX, recover_trX, teX, recover_teX]


[trX, re_trX, teX, re_teX] = pca_dataset(train_dataset, test_dataset, 2)
print('PCA reconstruction loss on Training data: %f '
      % reconstruct_error(train_dataset, re_trX))
print('PCA reconstruction loss on Test data: %f '
      % reconstruct_error(test_dataset, re_teX))
plot_2d_scatter(trX, trY, 'PCA on Training')
plot_2d_scatter(teX, teY, 'PCA on Testing')


def autoencode_dataset(train_dataset, test_dataset):
    feature_size = train_dataset.shape[1]
    encoder_size = 2
    autoencoder = SparseAutoencoder(
        feature_size, encoder_size, sparsity=0.05,
        sparsity_weight=1, encode_lr=0.001, l2_weight=0.001)
    batch_size = 1000
    num_steps = 50000
    autoencoder.train(train_dataset, batch_size, num_steps)
    test_loss = autoencoder.calc_reconstruct_loss(test_dataset)
    print("Testset decode loss: %f" % test_loss)
    encoded_train_dataset = autoencoder.encode_dataset(train_dataset)
    encoded_test_dataset = autoencoder.encode_dataset(test_dataset)
    return [encoded_train_dataset, encoded_test_dataset]


[encoded_train_dataset, encoded_test_dataset] = autoencode_dataset(
    train_dataset, test_dataset)
plot_2d_scatter(encoded_train_dataset, trY, 'Autoencoder on Training')
plot_2d_scatter(encoded_test_dataset, teY, 'Autoencoder on Testing')


def rbm_dataset(train_dataset, test_dataset):
    feature_size = train_dataset.shape[1]
    num_hidden_rbm = 2
    rbm_lr = 0.01
    batch_size = 1000
    num_steps = 4000
    rbm = RestrictedBoltzmannMachine(feature_size, num_hidden_rbm,
                                     batch_size, rbm_lr)
    print('Restricted Boltzmann Machine built')
    rbm.train(train_dataset, batch_size, num_steps)
    rbm.test_reconstruction(test_dataset)
    # Encode datasets
    hrand = rand(train_dataset.shape[0], num_hidden_rbm)

    encoded_train_dataset = rbm.encode_dataset(train_dataset, hrand)
    print(encoded_train_dataset[5:10, :])
    hrand = rand(test_dataset.shape[0], num_hidden_rbm)

    encoded_test_dataset = rbm.encode_dataset(test_dataset, hrand)
    print(encoded_test_dataset[5:10, :])
    return [encoded_train_dataset, encoded_test_dataset]


# [encoded_train_dataset, encoded_test_dataset] = rbm_dataset(
    # train_dataset, test_dataset)
# plot_2d_scatter(encoded_train_dataset, trY, 'RBM on Training')
# plot_2d_scatter(encoded_test_dataset, teY, 'RBM on Testing')

plt.show()


def plot_3d_scatter(X, y):
    fig = plt.figure(1, figsize=(5, 4))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    for (i, name) in enumerate(label_names):
        ax.text3D(X[y == i, 0].mean(),
                  X[y == i, 1].mean() + 1.5,
                  X[y == i, 2].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        ax.scatter(X[y == i, 0], X[y == i, 1], X[y == i, 2],
                   c=colors[i], marker=markers[i], lw=0.0,
                   label=name)
    # Reorder the labels to have colors matching the cluster results
    # y = np.choose(y, [0, 1, 2, 3, 4]).astype(np.float)
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.spectral)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    plt.show()
