from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input
from keras.layers.embeddings import Embedding

from preprocess.unsw import get_feature_names, discovery_feature_volcabulary
from preprocess.unsw import discovery_integer_map, discovery_continuous_map
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import pandas as pd


def get_dataset(dataset_name, headers):
    df = pd.read_csv(dataset_name, names=headers, sep=',',
                     skipinitialspace=True, skiprows=1, engine='python')
    X = df.drop('attack_cat', axis=1)
    labels = df['label'].astype(int).as_matrix()
    num_classes = 2
    y = np.zeros(shape=(labels.shape[0], num_classes))
    for (i, l) in enumerate(labels):
        y[i, l] = 1

    return X, y


def embedding_feature(X, y, X_test, y_test, name):
    vocabulary_dim = int(max(np.amax(X), np.amax(X_test)) + 1)
    embedding_dim = int(np.ceil(np.log2(vocabulary_dim)))
    embedding_dim = min(4, embedding_dim)

    print("|V| =", vocabulary_dim)
    print("|E| =", embedding_dim)
    print("X.shape =", X.shape)
    feature = Input(shape=(1, ), name='feature_%s' % name)
    embedding = Embedding(vocabulary_dim, embedding_dim,
                          input_length=1)(feature)
    flatten = Flatten()(embedding)
    sm = Dense(2, activation='softmax', name='output')(flatten)
    model = Model(inputs=feature, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(X, y, epochs=8, batch_size=100, shuffle=True)
    scores = model.evaluate(X_test, y_test, batch_size=X_test.shape[0])
    print(scores)


def min_max_feature(X, y, X_test, y_test, name):
    mm = MinMaxScaler()
    mm.fit(np.concatenate((X, X_test), axis=0))
    X = mm.transform(X)
    X_test = mm.transform(X_test)

    feature = Input(shape=(X.shape[1], ),  name='feature_%s' % name)
    sm = Dense(2, activation='softmax', name='output')(feature)
    model = Model(inputs=feature, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    model.fit(X, y, epochs=8, batch_size=100, shuffle=True)
    scores = model.evaluate(X_test, y_test, batch_size=X_test.shape[0])
    print(scores)


dataset_names = ['UNSW/UNSW_NB15_%s-set.csv' % x
                 for x in ['training', 'testing']]
feature_file = 'UNSW/feature_names_train_test.csv'

symbolic_features = discovery_feature_volcabulary(dataset_names)
integer_features = discovery_integer_map(feature_file, dataset_names)
headers, _, _, _ = get_feature_names(feature_file)

X, y = get_dataset(dataset_names[0], headers)
test_X, test_y = get_dataset(dataset_names[1], headers)

for (name, values) in symbolic_features.items():
    raw_X = X[name].as_matrix()
    raw_X_test = test_X[name].as_matrix()
    le = LabelEncoder()
    le.fit(np.concatenate((raw_X, raw_X_test), axis=0))
    temp = le.transform(raw_X)
    temp_test = le.transform(raw_X_test)
    feature = np.reshape(temp, (temp.shape[0], 1))
    feature_test = np.reshape(temp_test, (temp_test.shape[0], 1))
    min_max_feature(feature, y, feature_test, test_y, name)
    # embedding_feature(feature, y, feature_test, test_y, name)
