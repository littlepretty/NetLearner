from __future__ import print_function, division
import numpy as np
from netlearner.utils import min_max_scale, measure_prediction
from netlearner.utils import permutate_dataset
from netlearner.rbm import RestrictedBoltzmannMachine
from preprocess.unsw import generate_dataset
import tensorflow as tf
from math import ceil
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import pickle
import os


def plot_history(train_loss, valid_loss, test_loss, fig_dir):
    fig, ax1 = plt.subplots()
    ln1 = ax1.plot(train_loss, 'r--', label='Train')
    ln2 = ax1.plot(valid_loss, 'b:', label='Valid')
    ax1.set_ylabel('Train/Valid Loss', color='r')

    ax2 = ax1.twinx()
    ln3 = ax2.plot(test_loss, 'g-.', label='Test')
    ax2.set_ylabel('Test Loss', color='g')

    lns = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper left')

    ax1.grid(color='k', linestyle=':', linewidth=1)
    ax2.grid(color='k', linestyle=':', linewidth=1)
    fig.tight_layout()
    plt.savefig(fig_dir + 'history.pdf', format='pdf')
    plt.close()


def pretrain_model():
    num_hidden_rbm = 800
    rbm_lr = 0.01
    batch_size = 10
    num_epoch = 160
    num_steps = ceil(num_samples / batch_size * num_epoch)
    rbm = RestrictedBoltzmannMachine(feature_size, num_hidden_rbm,
                                     batch_size, trans_func=tf.nn.sigmoid,
                                     num_labels=num_classes, dirname=data_dir)
    rbm.train_with_labels(X, y, int(num_steps), X_valid, rbm_lr)
    test_loss = rbm.calc_reconstruct_loss(X_test)
    print("Testset reconstruction error: %f" % test_loss)
    rbm_w = rbm.get_weights('w')
    rbm_b = rbm.get_weights('bh')
    tf.reset_default_graph()

    input_layer = Input(shape=(feature_size, ), name='input')
    h1 = Dense(num_hidden_rbm, activation='sigmoid', name='h1')(input_layer)
    h1 = Dropout(0.8)(h1)
    h2 = Dense(480, activation='sigmoid', name='h2')(h1)
    sm = Dense(num_classes, activation='softmax', name='output')(h2)
    mlp = Model(inputs=input_layer, outputs=sm, name='rbm_mlp')
    mlp.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    mlp.summary()
    mlp.get_layer('h1').set_weights([rbm_w, rbm_b])
    mlp.save(pretrained_mlp_path)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model_dir = 'RBM/'
generate_dataset(binary_label=True, one_hot_encoding=True, root_dir=model_dir)
data_dir = model_dir + 'UNSW/'
pretrained_mlp_path = data_dir + 'rbm_mlp.h5'

raw_train_dataset = np.load(data_dir + 'train_dataset.npy')
raw_valid_dataset = np.load(data_dir + 'valid_dataset.npy')
y = np.load(data_dir + 'train_labels.npy')
raw_test_dataset = np.load(data_dir + 'test_dataset.npy')
y_test = np.load(data_dir + 'test_labels.npy')
[X, X_valid, X_test] = min_max_scale(raw_train_dataset, raw_valid_dataset,
                                     raw_test_dataset)
X, y = permutate_dataset(X, y)
print('Training set', X.shape, y.shape)
print('Test set', X_test.shape)
(num_samples, num_classes) = y.shape
feature_size = X.shape[1]
num_epoch = 240
batch_size = 96
weights = None
pretrain_model()
fold = 5
skf = StratifiedKFold(n_splits=fold)
hist = {'train_loss': [], 'valid_loss': []}
train_loss, valid_loss = [], []
y_flatten = np.argmax(y, axis=1)
for train_index, valid_index in skf.split(X, y_flatten):
    train_dataset, valid_dataset = X[train_index], X[valid_index]
    train_labels, valid_labels = y[train_index], y[valid_index]
    mlp = load_model(pretrained_mlp_path)
    history = mlp.fit(train_dataset, train_labels, batch_size, num_epoch,
                      verbose=1, class_weight=weights,
                      validation_data=(valid_dataset, valid_labels))
    score = mlp.evaluate(X_test, y_test, y_test.shape[0])
    print('Submodel test score: %s = %s' % (mlp.metrics_names, score))
    train_loss.append(history.history['loss'])
    valid_loss.append(history.history['val_loss'])

hist['train_loss'] = np.mean(train_loss, axis=0)
hist['valid_loss'] = np.mean(valid_loss, axis=0)
opt_epochs = np.argmin(hist['valid_loss'])
print('Optimal #Epochs:', opt_epochs + 1)
hist['opt_epochs'] = opt_epochs + 1

mlp = load_model(pretrained_mlp_path)
history = mlp.fit(X, y, batch_size, num_epoch,
                  verbose=1, class_weight=weights,
                  validation_data=(X_test, y_test))
predicted = mlp.predict(X_test, X_test.shape[0])
measure_prediction(predicted, y_test, data_dir, 'Test')

hist['test_loss'] = history.history['val_loss']
hist['test_acc_report'] = history.history['val_acc'][opt_epochs]
hist['test_acc'] = history.history['val_acc']
print('Test accuracy = %s' % hist['test_acc_report'])
output = open(data_dir + '%dFold%d.pkl' % (fold, num_epoch), 'wb')
pickle.dump(hist, output)
output.close()
"""
filename = open(data_dir + '%dFold%d.pkl' % (fold, num_epochs), 'rb')
hist = pickle.load(filename)
"""
plot_history(hist['train_loss'], hist['valid_loss'], hist['test_loss'],
             data_dir)
