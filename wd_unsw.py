from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from preprocess.unsw import get_feature_names, discovery_feature_volcabulary
from preprocess.unsw import generate_header  # , discovery_discrete_range


def build_model(model_dir, model_type):
    hidden_layers = [400, 512, 640]
    if model_type == 'wide':
        m = tf.estimator.LinearClassifier(
            model_dir=model_dir, feature_columns=wide_columns)
    elif model_type == 'deep':
        m = tf.estimator.DNNClassifier(
            model_dir=model_dir, feature_columns=deep_columns,
            hidden_units=hidden_layers)
    else:
        m = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=hidden_layers)
    print('Hidden units in each layer:', hidden_layers)
    return m


def augment_dataset(filename, output='train'):
    print('dealing with %s' % filename)
    df = pd.read_csv(filename,
                     names=CSV_COLUMNS,
                     sep=',',
                     skipinitialspace=True,
                     skiprows=1,
                     engine='python')

    df = df.drop('attack_cat', axis=1)
    labels = df['label'].astype(int)

    numeric = df[continuous_names + discrete_names].as_matrix()
    symbolic = df[symbolic_names].as_matrix()

    transformer = QuantileTransformer()
    augment = transformer.fit_transform(numeric)
    scaler = MinMaxScaler()
    numeric = scaler.fit_transform(numeric)

    combined = np.concatenate((symbolic, numeric, augment), axis=1)
    full_columns = symbolic_names + continuous_names + discrete_names \
        + quantile_names
    combined_df = pd.DataFrame(combined, columns=full_columns,
                               index=labels.index.tolist())
    temp = pd.concat([combined_df, labels], axis=1)
    name = 'UNSW/%s_temp.csv' % output
    temp.to_csv(name, index=False)

    return name, full_columns + ['label']


def input_builder(filename, full_columns, num_epochs, shuffle):
    print('dealing with %s' % filename)
    df = pd.read_csv(filename,
                     names=full_columns,
                     sep=',',
                     skipinitialspace=True,
                     skiprows=1,
                     engine='python')
    labels = df['label'].astype(int)
    dataset = df.drop('label', axis=1)
    print('Raw dataset shape:', dataset.shape)
    print('Raw label shape:', labels.shape)

    return tf.estimator.inputs.pandas_input_fn(
        x=dataset, y=labels, batch_size=160, num_epochs=num_epochs,
        shuffle=shuffle, num_threads=1)


def train_and_eval(model_dir, model_type, train_steps,
                   train_filenames, test_filename, full_columns):
    m = build_model(model_dir, model_type)
    m.train(
        input_fn=input_builder(train_filenames, full_columns,
                               num_epochs=None, shuffle=False),
        steps=train_steps)

    results = m.evaluate(
        input_fn=input_builder(test_filename, full_columns,
                               num_epochs=1, shuffle=False),
        steps=None)
    for key in results:
        print("%s: %s" % (key, results[key]))


train_filename = 'UNSW/UNSW_NB15_training-set.csv'
test_filename = 'UNSW/UNSW_NB15_testing-set.csv'
feature_filename = 'UNSW/feature_names_train_test.csv'
CSV_COLUMNS, symbolic_names, continuous_names, discrete_names = \
    get_feature_names(feature_filename)
quantile_names = []
for name in continuous_names + discrete_names:
    quantile_names.append(name + '_quantile')

print(symbolic_names, len(symbolic_names))
print(continuous_names, len(continuous_names))
print(discrete_names, len(discrete_names))
print(quantile_names, len(quantile_names))

header = generate_header(CSV_COLUMNS)
print('Headers for dataset file:', header)

symbolic_features = discovery_feature_volcabulary([train_filename,
                                                   test_filename])
symbolic_columns = dict()
for (name, categorical_values) in symbolic_features.items():
    column = tf.feature_column.categorical_column_with_vocabulary_list(
        name, categorical_values)
    symbolic_columns[name] = column

continuous_columns = dict()
for name in continuous_names:
    column = tf.feature_column.numeric_column(name)
    continuous_columns[name] = column

for name in discrete_names:
    column = tf.feature_column.numeric_column(name)
    continuous_columns[name] = column

for name in quantile_names:
    column = tf.feature_column.numeric_column(name)
    continuous_columns[name] = column

# convert discrete features into categorical columns
discrete_columns = dict()
"""
upper, lower = discovery_discrete_range(filenames,
discrete_names, CSV_COLUMNS)
for name in discrete_names:
    column = tf.feature_column.categorical_column_with_identity(
    name, upper[name] - lower[name] + 1)
    discrete_columns[name] = column
    """

# Build components for the wide model
base_columns = symbolic_columns.values() + discrete_columns.values()
cross_columns = [
    tf.feature_column.crossed_column(
        ['proto', 'service'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['proto', 'state'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['service', 'state'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['proto', 'service', 'state'], hash_bucket_size=8000)
]
wide_columns = base_columns + cross_columns
print('#wide components:', len(wide_columns))

# Build components for the deep model
indicator_columns = []  # low dimension categorical features
for name in symbolic_names:  # ['state', 'service']
    column = symbolic_columns[name]
    indicator_columns.append(tf.feature_column.indicator_column(column))

print('indicator columns', len(indicator_columns))

"""
low_discrete_names = ['trans_depth', 'ct_state_ttl',
'ct_flw_http_mthd', 'ct_ftp_cmd']
for name in low_discrete_names:
    column = discrete_columns[name]
    indicator_columns.append(tf.feature_column.indicator_column(column))
    """

# convert high dimension categorical features to embeddings
embedding_columns = []
for (name, column) in symbolic_columns.items():
    volcabulary_size = len(symbolic_features[name])
    print(name, '|V| =', volcabulary_size)
    dim = np.ceil(np.log2(volcabulary_size))
    embedding = tf.feature_column.embedding_column(column, dim)
    embedding_columns.append(embedding)

print('embedding columns', len(embedding_columns))
"""
high_discrete_names = set(discrete_names).difference(set(low_discrete_names))
for name in high_discrete_names:
volcabulary_size = upper[name] - lower[name] + 1
# print(name, '|V| =', volcabulary_size)
dim = 4
column = discrete_columns[name]
embedding = tf.feature_column.embedding_column(column, dim)
embedding_columns.append(embedding)
"""
deep_columns = indicator_columns + embedding_columns \
    + continuous_columns.values()
print('#deep components:', len(deep_columns))

train_steps = 160000
# train_temp, full_columns = augment_dataset(train_filename)
# test_temp, full_columns = augment_dataset(test_filename, 'test')
full_columns = symbolic_names + continuous_names + \
    discrete_names + quantile_names + ['label']
train_temp = 'UNSW/train_temp.csv'
test_temp = 'UNSW/test_temp.csv'
model_dir = None  # 'WideDeepModel'
train_and_eval(model_dir, 'wide+deep', train_steps,
               train_temp, test_temp, full_columns)
