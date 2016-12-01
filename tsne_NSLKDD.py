import numpy as np
import matplotlib.pyplot as plt
from netlearner.utils import min_max_normalize
from netlearner.tsne import tsne
import pickle


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

total_size = test_dataset.shape[0]
sample_index = np.random.choice(total_size,
                                size=int(0.96 * total_size), replace=False)
test_dataset = test_dataset[sample_index, :]
test_labels = np.argmax(test_labels[sample_index], 1)

test_Y = tsne(test_dataset, 2, test_dataset.shape[1])
fig, ax = plt.subplots(1, 1)
ax.scatter(test_Y[:, 0], test_Y[:, 1], 20, test_labels)
pickle.dump(fig, open('origin_data.pickle', 'wb'))
