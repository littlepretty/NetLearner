import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle
from netlearner.utils import min_max_normalize


def tsne_dataset(dataset, labels, output):
    total_size = dataset.shape[0]
    sample_index = np.random.choice(total_size,
                                    size=int(0.2 * total_size), replace=False)
    dataset = dataset[sample_index, :]
    labels = np.argmax(labels[sample_index], 1)

    tsne = TSNE(n_components=2, init='pca', n_iter=2000)
    Y = tsne.fit_transform(dataset)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pickle.dump(fig, open(output, 'wb'))


test_dataset = np.load('testset.rbm.npy')
test_labels = np.load('NSLKDD/test_ref.npy')
output = 'rbm_encoded.pickle'
tsne_dataset(test_dataset, test_labels, output)

raw_train_dataset = np.load('NSLKDD/train_dataset.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
raw_valid_dataset = np.load('NSLKDD/valid_dataset.npy')
valid_labels = np.load('NSLKDD/valid_ref.npy')
raw_test_dataset = np.load('NSLKDD/test_dataset.npy')
[train_dataset, valid_dataset, test_dataset] = min_max_normalize(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
output = 'raw.pickle'
tsne_dataset(test_dataset, test_labels, output)
