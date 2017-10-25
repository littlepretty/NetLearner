from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.embeddings import Embedding
# from keras import optimizers
from preprocess.unsw import generate_dataset
from netlearner.utils import quantile_transform
import numpy as np

generate_dataset(one_hot_encode=False)
raw_train_dataset = np.load('UNSW/train_dataset.npy')
train_labels = np.load('UNSW/train_labels.npy')
raw_valid_dataset = np.load('UNSW/valid_dataset.npy')
valid_labels = np.load('UNSW/valid_labels.npy')
raw_test_dataset = np.load('UNSW/test_dataset.npy')
test_labels = np.load('UNSW/test_labels.npy')

embedded_features = raw_train_dataset[:, -3:]
print(embedded_features.shape)
vocabulary_dim = int(np.amax(embedded_features)) + 1
embedding_dim = int(np.log2(vocabulary_dim)) + 1
num_features = embedded_features.shape[1]
print("|V| =", vocabulary_dim)
print("|E| =", embedding_dim)
print("|F| =", num_features)

model1 = Sequential()
model1.add(Embedding(vocabulary_dim, embedding_dim, input_length=num_features))
model1.add(Flatten())
model1.compile('rmsprop', 'mse')
train_embeddings = model1.predict(embedded_features)
valid_embeddings = model1.predict(raw_valid_dataset[:, -3:])
test_embeddings = model1.predict(raw_test_dataset[:, -3:])
print(train_embeddings.shape)
print(test_embeddings.shape)

columns = np.array(range(1, 6) + range(8, 16) + range(17, 19) +
                   range(23, 25) + [26])
[train_dataset, valid_dataset, test_dataset] = quantile_transform(
    raw_train_dataset[:, :-3],
    raw_valid_dataset[:, :-3],
    raw_test_dataset[:, :-3], columns)

X_train = np.concatenate((train_dataset, train_embeddings), axis=1)
X_valid = np.concatenate((valid_dataset, valid_embeddings), axis=1)
X_test = np.concatenate((test_dataset, test_embeddings), axis=1)
print(X_train.shape, X_test.shape)

num_features = X_train.shape[1]
num_classes = train_labels.shape[1]

model2 = Sequential()
model2.add(Dense(400, input_dim=num_features))
model2.add(Activation('relu'))
model2.add(Dropout(0.8))
model2.add(Dense(512))
model2.add(Activation('relu'))
model2.add(Dropout(0.8))
model2.add(Dense(640))
model2.add(Activation('relu'))
model2.add(Dense(num_classes))
model2.add(Activation('softmax'))

# adam = optimizers.Adam(lr=0.001, decay=0.002)
model2.compile(loss='categorical_crossentropy',
               optimizer='adam',
               metrics=['accuracy'])
history = model2.fit(X_train, train_labels,
                     batch_size=100,
                     epochs=160,
                     verbose=1,
                     validation_data=(X_valid, valid_labels))
score = model2.evaluate(X_test, test_labels, batch_size=100, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
