from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from netlearner.utils import min_max_scale
from netlearner.utils import measure_prediction
from preprocess import nslkdd
from keras.regularizers import l2
from keras.models import Model
from keras.utils import multi_gpu_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import StratifiedKFold
import os
import pickle
import matplotlib.pyplot as plt


def build_model():
    with tf.device("/cpu:0"):
        il = Input(shape=(feature_size, ), name='input')
        h1 = Dense(hidden_size[0], name='h1', kernel_regularizer=l2(beta))(il)
        h1 = Activation('relu')(h1)
        h1 = Dropout(keep_prob)(h1)
        h2 = Dense(hidden_size[1], name='h2', kernel_regularizer=l2(beta))(h1)
        h2 = BatchNormalization()(h2)
        h2 = Activation('sigmoid')(h2)
        h2 = Dropout(keep_prob)(h2)
        sm = Dense(num_classes, name='output')(h2)
        sm = Activation('softmax')(sm)
        mlp = Model(inputs=il, outputs=sm, name='mlp')

    mlp = multi_gpu_model(mlp, gpus=4)
    mlp.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    mlp.summary()
    return mlp


def plot_history(train_loss, valid_loss, test_loss, fig_dir):
    fig, ax1 = plt.subplots()
    ln1 = ax1.plot(train_loss / (fold - 1.0), 'r--', label='Trainset')
    ln2 = ax1.plot(valid_loss, 'b:', label='Validset')
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


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
model_dir = 'KerasMLP/'
data_dir = model_dir + 'NSLKDD/'

batch_size = 64
keep_prob = 0.80
num_epochs = 120
beta = 0.0000
hidden_size = [800, 480]
fold = 5
weights = None  # {0: 1.0, 1: 3.0, 2: 1.0, 3: 8.0, 4: 3.0}

nslkdd.generate_dataset(False, True, model_dir)
raw_train_dataset = np.load(data_dir + 'train_dataset.npy')
raw_test_dataset = np.load(data_dir + 'test_dataset.npy')
y = np.load(data_dir + 'train_labels.npy')
y_test = np.load(data_dir + 'test_labels.npy')
X, _, X_test = min_max_scale(raw_train_dataset, None, raw_test_dataset)
y_flatten = np.argmax(y, axis=1)
print('Train dataset', X.shape, y.shape, y_flatten.shape)
print('Test dataset', X_test.shape, y_test.shape)
num_samples, num_classes = y.shape
feature_size = X.shape[1]

skf = StratifiedKFold(n_splits=fold)
hist = {'train_loss': [], 'valid_loss': []}
train_loss, valid_loss = [], []
train_accu, valid_accu = [], []
for train_index, valid_index in skf.split(X, y_flatten):
    train_dataset, valid_dataset = X[train_index], X[valid_index]
    train_labels, valid_labels = y[train_index], y[valid_index]
    mlp = build_model()
    history = mlp.fit(train_dataset, train_labels, batch_size, num_epochs,
                      verbose=1, class_weight=weights,
                      validation_data=(valid_dataset, valid_labels))
    score = mlp.evaluate(X_test, y_test, y_test.shape[0])
    print('Submodel test score: %s = %s' % (mlp.metrics_names, score))
    train_loss.append(history.history['loss'])
    valid_loss.append(history.history['val_loss'])
    train_accu.append(history.history['acc'])
    valid_accu.append(history.history['val_acc'])


hist['train_loss'] = np.mean(train_loss, axis=0)
hist['valid_loss'] = np.mean(valid_loss, axis=0)
hist['train_accu'] = np.mean(train_accu, axis=0)
hist['valid_accu'] = np.mean(valid_accu, axis=0)

opt_epochs = np.argmin(hist['valid_loss'])
print('Optimal #Epochs:', opt_epochs + 1)
hist['opt_epochs'] = opt_epochs + 1

mlp = build_model()
history = mlp.fit(X, y, batch_size, num_epochs,
                  verbose=1, class_weight=weights,
                  validation_data=(X_test, y_test))
predicted = mlp.predict(X_test, X_test.shape[0])
measure_prediction(predicted, y_test, data_dir, 'Test')

hist['test_loss'] = history.history['val_loss']
hist['test_acc_report'] = history.history['val_acc'][opt_epochs]
hist['test_acc'] = history.history['val_acc']
print('Test accuracy = %s' % hist['test_acc_report'])
output = open(data_dir + '%dFold%d.pkl' % (fold, num_epochs), 'wb')
pickle.dump(hist, output)
output.close()
"""
filename = open(data_dir + '%dFold%d.pkl' % (fold, num_epochs), 'rb')
hist = pickle.load(filename)
"""
plot_history(hist['train_loss'], hist['valid_loss'], hist['test_loss'],
             data_dir)
