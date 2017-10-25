from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from preprocess.full_unsw import get_feature_names, symbolic_features
from preprocess.full_unsw import generate_header  # , discovery_discrete_range


CSV_COLUMNS, symbolic_names, continuous_names, discrete_names = \
    get_feature_names('UNSW/feature_names.csv')
print(symbolic_names, len(symbolic_names))
print(continuous_names, len(continuous_names))
print(discrete_names, len(discrete_names))
header = generate_header(CSV_COLUMNS)
print('Headers for dataset file:', header)

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
cross_columns = []
wide_columns = base_columns + cross_columns
print('#wide components:', len(wide_columns))

# Build components for the deep model
indicator_columns = []  # low dimension categorical features
for name in ['state', 'service']:
    column = symbolic_columns[name]
    indicator_columns.append(tf.feature_column.indicator_column(column))

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
    dim = 4
    embedding = tf.feature_column.embedding_column(column, dim)
    embedding_columns.append(embedding)
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


def build_model(model_dir, model_type):
    if model_type == 'wide':
        m = tf.estimator.LinearClassifier(
            model_dir=model_dir, feature_columns=wide_columns)
    elif model_type == 'deep':
        m = tf.estimator.DNNClassifier(
            model_dir=model_dir, feature_columns=deep_columns,
            hidden_units=[100])
    else:
        m = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[100])

    return m


def fill_nan(df):
    for name in symbolic_names:
        df[name].fillna('', inplace=True)

    for name in continuous_names:
        df[name].fillna(0.0, inplace=True)

    for name in discrete_names:
        df[name].fillna(0, inplace=True, downcast='infer')


def input_builder(filenames, num_epochs, shuffle):
    data_frames = []
    label_frames = []

    for filename in filenames:
        print('dealing with %s' % filename)
        df = pd.read_csv(filename,
                         names=CSV_COLUMNS,
                         sep=',',
                         skipinitialspace=True,
                         engine='python',
                         na_values='-',
                         nrows=100000)
        # fill_nan(df)
        df.dropna(axis=0, how='any', inplace=True)
        label = df['label'].astype(int)
        data = df.drop('attack_cat', axis=1)
        data = data.drop('label', axis=1)
        for name in discrete_names:
            data[name] = pd.to_numeric(data[name], errors='raise',
                                       downcast='signed')
            # print(name, data[name].dtype)

        data_frames.append(data)
        label_frames.append(label)

    dataset = pd.concat(data_frames)
    labels = pd.concat(label_frames)
    print('Raw dataset shape:', dataset.shape)
    print('Raw label shape:', labels.shape)
    return tf.estimator.inputs.pandas_input_fn(
        x=dataset, y=labels, batch_size=40, num_epochs=num_epochs,
        shuffle=shuffle, num_threads=1)


def train_and_eval(model_dir, model_type, train_steps,
                   train_filenames, test_filename):
    m = build_model(model_dir, model_type)
    m.train(
        input_fn=input_builder(train_filenames, num_epochs=None, shuffle=False),
        steps=train_steps)

    results = m.evaluate(
        input_fn=input_builder([test_filename], num_epochs=1, shuffle=False),
        steps=None)
    for key in results:
        print("%s: %s" % (key, results[key]))


filenames = ['UNSW/UNSW-NB15_%d.csv' % x for x in range(1, 3)]
train_filenames = filenames[:-1]
test_filename = filenames[-1]
print(test_filename)
train_steps = 1
model_dir = None  # 'WideDeepModel'
train_and_eval(model_dir, 'wide+deep', train_steps,
               train_filenames, test_filename)
