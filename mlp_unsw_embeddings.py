from keras.models import Sequential
from keras.layers.core import Flatten
from keras.layers.embeddings import Embedding
from preprocess.unsw import generate_dataset
from netlearner.utils import min_max_scale
from netlearner.utils import hyperparameter_summary
from netlearner.utils import augment_quantiled, permutate_dataset
from netlearner.multilayer_perceptron import MultilayerPerceptron
import numpy as np
import tensorflow as tf


def embedding_symbolic_feature(X_train, X_valid, X_test):
    vocabulary_dim = int(max(np.amax(X_train), np.amax(X_test)) + 1)
    embedding_dim = int(np.ceil(np.log2(vocabulary_dim)))
    print("|V| =", vocabulary_dim)
    print("|E| =", embedding_dim)

    model = Sequential()
    model.add(Embedding(vocabulary_dim, embedding_dim, input_length=1))
    model.add(Flatten())
    model.compile('rmsprop', 'mse')
    e_train = model.predict(X_train)
    e_valid = model.predict(X_valid)
    e_test = model.predict(X_test)
    print(e_train.shape)
    # print(np.amax(e_train, axis=0), np.amin(e_train, axis=0))
    print(e_test.shape)
    # print(np.amax(e_test, axis=0), np.amin(e_test, axis=0))
    return e_train, e_valid, e_test


generate_dataset(one_hot_encode=False)
raw_train = np.load('UNSW/train_dataset.npy')
y_train = np.load('UNSW/train_labels.npy')
raw_valid = np.load('UNSW/valid_dataset.npy')
y_valid = np.load('UNSW/valid_labels.npy')
raw_test = np.load('UNSW/test_dataset.npy')
y_test = np.load('UNSW/test_labels.npy')

train_cont = raw_train[:, :-3]
valid_cont = raw_valid[:, :-3]
test_cont = raw_test[:, :-3]
train_disc = raw_train[:, -3:]
valid_disc = raw_valid[:, -3:]
test_disc = raw_test[:, -3:]

print("Continuous dataset", train_cont.shape)
columns = np.array(range(1, 6) + range(8, 16) + range(17, 19) +
                   range(23, 25) + [26])
[X_train, X_valid, X_test] = augment_quantiled(train_cont,
                                               valid_cont,
                                               test_cont, columns)
print("Augmenting quantiled dataset", X_train.shape)

for i in range(3):
    [ftr, fv, fte] = embedding_symbolic_feature(train_disc[:, i],
                                                valid_disc[:, i],
                                                test_disc[:, i])
    X_train = np.concatenate((X_train, ftr), axis=1)
    print(X_train.shape)
    print(X_valid.shape, fv.shape)
    X_valid = np.concatenate((X_valid, fv), axis=1)
    X_test = np.concatenate((X_test, fte), axis=1)

X_train = np.concatenate((X_train, train_disc), axis=1)
X_valid = np.concatenate((X_valid, valid_disc), axis=1)
X_test = np.concatenate((X_test, test_disc), axis=1)

print("Augmenting discrete & embedding dataset", X_train.shape)
[X_train, X_valid, X_test] = min_max_scale(X_train, X_valid, X_test)
print("Min-max scaled dataset", X_train.shape, X_test.shape)

X_train, y_train = permutate_dataset(X_train, y_train)
X_valid, y_valid = permutate_dataset(X_valid, y_valid, 'Valid')
X_test, y_test = permutate_dataset(X_test, y_test, 'Test')

num_samples, num_features = X_train.shape
num_classes = y_train.shape[1]
batch_size = 40
keep_prob = 0.80
beta = 0.0001
weights = [1.0, 1.0]
num_epochs = [60]
init_lrs = [0.001]
hidden_layer_sizes = [
                      [480, 512, 640],
                      # [800, 640], [160, 80], [80, 40],
                      # [400, 360, 320],
                      # [160, 120, 80], [120, 80, 40],
                      ]
for hidden_layer_size in hidden_layer_sizes:
    for init_lr in init_lrs:
        for num_epoch in num_epochs:
            num_steps = int(num_samples / batch_size * num_epoch)
            decay_steps = num_steps / num_epoch
            mp_classifier = MultilayerPerceptron(num_features,
                                                 hidden_layer_size,
                                                 num_classes, init_lr,
                                                 decay_steps, beta,
                                                 tf.nn.relu,
                                                 tf.nn.l2_loss, weights,
                                                 tf.train.AdamOptimizer,
                                                 name='PureMLP-UNSW2C-Embed')
            mp_classifier.train_with_labels(X_train, y_train,
                                            batch_size, num_steps,
                                            X_valid, y_valid,
                                            X_test, y_test,
                                            keep_prob)
            hyperparameter = {'hidden_layer_size': hidden_layer_size,
                              'init_lr': init_lr,
                              'num_epochs': num_epoch,
                              'num_steps': num_steps,
                              'regularization beta': beta,
                              'optimizer': 'AdamOptimizer',
                              'keep_prob': keep_prob,
                              'act_func': 'RELU',
                              'class_weights': weights,
                              'batch_size': batch_size, }
            hyperparameter_summary(mp_classifier.dirname,
                                   hyperparameter)
            f = open(mp_classifier.dirname + '/test.log')
            print(f.read())
            f.close()
            mp_classifier.exit()
