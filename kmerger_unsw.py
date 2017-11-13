from keras.models import Model
from keras.layers import Dense, Input, concatenate, Flatten
from keras.layers import Embedding, BatchNormalization

from preprocess.unsw import get_feature_names, discovery_feature_volcabulary
from preprocess.unsw import discovery_integer_map, discovery_continuous_map
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import pandas as pd
import numpy as np


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


dataset_names = ['UNSW/UNSW_NB15_%s-set.csv' % x
                 for x in ['training', 'testing']]
feature_file = 'UNSW/feature_names_train_test.csv'

symbolic_features = discovery_feature_volcabulary(dataset_names)
integer_features = discovery_integer_map(feature_file, dataset_names)
headers, _, _, _ = get_feature_names(feature_file)

X, y = get_dataset(dataset_names[0], headers)
test_X, test_y = get_dataset(dataset_names[1], headers)

train_dict = dict()
test_dict = dict()
merged_dim = 0
merged_inputs = []

# Define embedding layers/inputs
embeddings = []

for (name, values) in symbolic_features.items():
    column = Input(shape=(1, ), name=name)
    merged_inputs.append(column)
    raw_data = X[name].as_matrix()
    test_raw_data = test_X[name].as_matrix()
    le = LabelEncoder()
    le.fit(np.concatenate((raw_data, test_raw_data), axis=0))
    train_dict[name] = le.transform(raw_data)
    test_dict[name] = le.transform(test_raw_data)

    dim_V = len(values)
    dim_E = int(min(7, np.ceil(np.log2(dim_V))))
    print('Dimension of %s E=%s and V=%s' % (name, dim_E, dim_V))
    temp = Embedding(output_dim=dim_E, input_dim=dim_V,
                     input_length=1, name='embed_%s' % name)(column)
    temp = Flatten(name='flat_%s' % name)(temp)
    embeddings.append(temp)
    merged_dim += dim_E

large_inputs = []
for (name, values) in integer_features.items():
    column = Input(shape=(1, ), name=name)
    merged_inputs.append(column)
    raw_data = X[name].astype('int64').as_matrix()
    test_raw_data = test_X[name].astype('int64').as_matrix()
    dim_V = int(values['max'] - values['min'] + 1)

    if dim_V < 8096:
        train_dict[name] = raw_data - values['min']
        test_dict[name] = test_raw_data - values['min']
        dim_E = int(min(5, np.ceil(np.log2(dim_V))))
        print('Dimension of %s E=%s and V=%s' % (name, dim_E, dim_V))
        temp = Embedding(output_dim=dim_E, input_dim=dim_V,
                         input_length=1, name='embed_%s' % name)(column)
        temp = Flatten(name='flat_%s' % name)(temp)
        embeddings.append(temp)
        merged_dim += dim_E
    else:
        large_inputs.append(column)
        print('Large feature %s is treated as continuous' % name)
        mm = MinMaxScaler()
        raw_data = raw_data.reshape((len(raw_data), 1))
        test_raw_data = test_raw_data.reshape((len(test_raw_data), 1))
        mm.fit(np.concatenate((raw_data, test_raw_data), axis=0))
        train_dict[name] = mm.transform(raw_data)
        test_dict[name] = mm.transform(test_raw_data)
        merged_dim += 1


continuous_features = discovery_continuous_map(feature_file, dataset_names)
continuous_inputs = Input(shape=(len(continuous_features), ),
                          name='continuous')
merged_inputs.append(continuous_inputs)
raw_data = X[continuous_features.keys()].as_matrix()
test_raw_data = test_X[continuous_features.keys()].as_matrix()
mm = MinMaxScaler()
mm.fit(np.concatenate((raw_data, test_raw_data), axis=0))
train_dict['continuous'] = mm.transform(raw_data)
test_dict['continuous'] = mm.transform(test_raw_data)
merged_dim += len(continuous_features)

print('merge input_dim for this dataset = %s' % merged_dim)
merge = concatenate(embeddings + large_inputs + [continuous_inputs],
                    name='merge_features')
h1 = Dense(400, activation='relu', name='hidden1')(merge)
# h2 = Dense(320, activation='relu', name='hidden2')(h1)
encode = BatchNormalization(name='unified_x')(h1)

h4 = Dense(640, activation='sigmoid', name='hidden_all')(encode)
sm = Dense(2, activation='softmax', name='output')(h4)

model = Model(inputs=merged_inputs, outputs=sm)
model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
history = model.fit(train_dict, {'output': y},
                    epochs=3, batch_size=80)
print(history.history)
score = model.evaluate(test_dict, test_y,
                       batch_size=test_y.shape[0], verbose=1)
print(score)
