from __future__ import print_function, division
import numpy as np
# import tensorflow as tf
from netlearner.utils import min_max_scale
# from netlearner.utils import permutate_dataset, measure_prediction
from preprocess.nslkdd import generate_dataset
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import StratifiedKFold
import os
import pickle
import matplotlib.pyplot as plt


def build_model():
    input_layer = Input(shape=(feature_size, ), name='input')
    h1 = Dense(hidden_size[0], activation='relu', name='h1')(input_layer)
    h1 = Dropout(keep_prob)(h1)
    h2 = Dense(hidden_size[1], activation='relu', name='h2')(h1)
    sm = Dense(num_classes, activation='softmax', name='output')(h2)
    mlp = Model(inputs=input_layer, outputs=sm, name='mlp')
    mlp.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    mlp.summary()
    return mlp


def plot_history(train_acc, valid_acc, test_acc, fig_dir):
    fig, ax1 = plt.subplots()
    l1 = ax1.plot(train_acc, 'r--', label='Train')
    l2 = ax1.plot(valid_acc, 'b:', label='Valid')
    ax1.set_ylabel('Train/Valid Accuracy', color='r')
    ax1.tick_params('y', colors='r')
    ax1.grid(color='k', linestyle=':', linewidth=1)

    ax2 = ax1.twinx()
    l3 = ax2.plot(test_acc, 'g-.', label='Test')
    ax2.set_ylabel('Test Accuracy', color='g')
    ax2.tick_params('y', colors='g')
    ax2.grid(color='k', linestyle=':', linewidth=1)

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)

    fig.tight_layout()
    plt.savefig(fig_dir + 'history.pdf', format='pdf')
    plt.close()


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
model_dir = 'KerasMLP/'
generate_dataset(False, True, model_dir)
data_dir = model_dir + 'NSLKDD/'
raw_train_dataset = np.load(data_dir + 'train_dataset.npy')
raw_test_dataset = np.load(data_dir + 'test_dataset.npy')
y = np.load(data_dir + 'train_labels.npy')
y_test = np.load(data_dir + 'test_labels.npy')

X, _, X_test = min_max_scale(raw_train_dataset, None, raw_test_dataset)
y_flatten = np.argmax(y, axis=1)
print('Train dataset', X.shape, y.shape, y_flatten.shape)
print('Test dataset', X_test.shape, y_test.shape)

batch_size = 80
keep_prob = 0.8
num_epochs = 240
num_samples, num_classes = y.shape
feature_size = X.shape[1]
hidden_size = [800, 480]
weights = {0: 0.05, 1: 0.15, 2: 0.05, 3: 0.6, 4: 0.15}
fold = 10
skf = StratifiedKFold(n_splits=fold)
hist = {'train_acc': [], 'valid_acc': []}
train_acc, valid_acc = [], []

for train_index, valid_index in skf.split(X, y_flatten):
    train_dataset, valid_dataset = X[train_index], X[valid_index]
    train_labels, valid_labels = y[train_index], y[valid_index]
    mlp = build_model()
    history = mlp.fit(train_dataset, train_labels, batch_size, num_epochs,
                      verbose=1, class_weight=weights,
                      validation_data=(valid_dataset, valid_labels))
    train_acc.append(history.history['acc'])
    valid_acc.append(history.history['val_acc'])

hist['train_acc'] = np.mean(train_acc, axis=0)
hist['valid_acc'] = np.mean(valid_acc, axis=0)
opt_epochs = np.argmax(hist['valid_acc'])
print('Optimal #Epochs:', opt_epochs + 1)
hist['opt_epochs'] = opt_epochs + 1

mlp = build_model()
history = mlp.fit(X, y, batch_size, num_epochs,
                  verbose=1, class_weight=weights,
                  validation_data=(X_test, y_test))
hist['test_loss'] = history.history['val_loss'][opt_epochs]
hist['test_acc_report'] = history.history['val_acc'][opt_epochs]
hist['test_acc'] = history.history['val_acc']
print('Test accuracy = %s' % hist['test_acc_report'])
plot_history(hist['train_acc'], hist['valid_acc'], hist['test_acc'], data_dir)
output = open(data_dir + '%dFold%d.pkl' % (fold, num_epochs), 'wb')
pickle.dump(hist, output)
output.close()
