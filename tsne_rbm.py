import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle


train_dataset = np.load('encoded_trainset.rbm.npy')
train_labels = np.load('NSLKDD/train_ref.npy')
test_dataset = np.load('encoded_testset.rbm.npy')
test_labels = np.load('NSLKDD/test_ref.npy')

total_size = test_dataset.shape[0]
sample_index = np.random.choice(total_size,
                                size=int(0.96 * total_size), replace=False)
test_dataset = test_dataset[sample_index, :]
test_labels = np.argmax(test_labels[sample_index], 1)

tsne = TSNE(n_components=2, init='pca', n_iter=2000)
test_Y = tsne.fit_transform(test_dataset)
fig, ax = plt.subplots(1, 1)
ax.scatter(test_Y[:, 0], test_Y[:, 1], 20, test_labels)
pickle.dump(fig, open('rbm_encoded.pickle', 'wb'))
