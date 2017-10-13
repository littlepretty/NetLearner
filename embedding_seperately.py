from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.embeddings import Embedding
# from keras import optimizers
from preprocess.unsw import generate_dataset
from netlearner.utils import augment_quantiled, min_max_scale, permutate_dataset
import numpy as np


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
X_valid, y_valid = permutate_dataset(X_valid, y_valid)
X_test, y_test = permutate_dataset(X_test, y_test)

num_features = X_train.shape[1]
num_classes = y_train.shape[1]

model2 = Sequential()
model2.add(Dense(400, input_dim=num_features, activation='relu'))
model2.add(Dropout(0.1))
model2.add(Dense(512, activation='relu'))
model2.add(Dropout(0.1))
model2.add(Dense(640, activation='relu'))
model2.add(Dense(num_classes, activation='softmax'))

# adam = optimizers.Adam(lr=0.001, decay=0.002)
model2.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
history = model2.fit(X_train, y_train,
                     batch_size=64,
                     epochs=60,
                     verbose=1,
                     class_weight={0: 1, 1: 1},
                     shuffle=True,
                     validation_data=(X_valid, y_valid))
score = model2.evaluate(X_test, y_test, batch_size=40, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
