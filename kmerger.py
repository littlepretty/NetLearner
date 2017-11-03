from keras.models import Model
from keras.layers import Dense, Input, concatenate, Flatten
from keras.layers import Embedding, BatchNormalization

from preprocess.unsw import get_feature_names, discovery_feature_volcabulary
from preprocess.unsw import discovery_integer_map, discovery_continuous_map
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

dataset_names = ['UNSW/UNSW_NB15_%s-set.csv' % x
                 for x in ['training', 'testing']]
feature_file = 'UNSW/feature_names_train_test.csv'

symbolic_features = discovery_feature_volcabulary(dataset_names)
integer_features = discovery_integer_map(feature_file, dataset_names)
headers, _, _, _ = get_feature_names(feature_file)
df = pd.read_csv(dataset_names[0],
                 names=headers, sep=',',
                 skipinitialspace=True, skiprows=1, engine='python')
X = df.drop('attack_cat', axis=1)
Y = df['label'].astype(int)
train_dict = dict()

# Define embedding layers/inputs
embeddings = []
embedding_inputs = []
for (name, values) in symbolic_features.items():
    column = Input(shape=(1, ), name=name)
    embedding_inputs.append(column)
    raw_data = X[name].as_matrix()
    le = LabelEncoder()
    le.fit(raw_data)
    train_dict[name] = le.transform(raw_data)

    dim_V = len(values)
    dim_E = int(min(7, np.ceil(np.log2(dim_V))))
    print('Dimension of E and V', dim_E, dim_V)
    temp = Embedding(output_dim=dim_E, input_dim=dim_V,
                     input_length=1, name='embed_%s' % name)(column)
    temp = Flatten(name='flat_%s' % name)(temp)
    embeddings.append(temp)

for (name, values) in integer_features.items():
    column = Input(shape=(1, ), name=name)
    embedding_inputs.append(column)
    train_dict[name] = X[name].astype('int64').as_matrix()

    dim_V = int(values['max'] - values['min'] + 1)
    dim_E = int(min(5, np.ceil(np.log2(dim_V))))
    print('Dimension of E and V', dim_E, dim_V)
    temp = Embedding(output_dim=dim_E, input_dim=dim_V,
                     input_length=1, name='embed_%s' % name)(column)
    temp = Flatten(name='flat_%s' % name)(temp)
    embeddings.append(temp)

continuous_features = discovery_continuous_map(feature_file, dataset_names)
print('symbolic %s' % symbolic_features)
print('integer %s' % integer_features)
print('continuous %s' % continuous_features)

continuous_inputs = Input(shape=(len(continuous_features), ),
                          name='continuous')
train_dict['continuous'] = X[continuous_features.keys()].as_matrix()

merge = concatenate(embeddings + [continuous_inputs], name='merged_input')
h1 = Dense(256, activation='relu', name='hidden1')(merge)
# h2 = Dense(320, activation='relu', name='hidden2')(h1)
encode = BatchNormalization(name='unified_x')(h1)

h4 = Dense(400, activation='sigmoid', name='hidden_all')(encode)
sm = Dense(1, activation='softmax', name='output')(h4)

print('input/output dict %s' % train_dict)
model = Model(inputs=embedding_inputs + [continuous_inputs], outputs=sm)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.summary()
model.fit(train_dict, {'output': Y.as_matrix()},
          epochs=1, batch_size=40)
