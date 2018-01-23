from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from preprocess.unsw import get_feature_names, discovery_feature_volcabulary
from preprocess.unsw import generate_header, discovery_discrete_range
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import pickle
import os


def build_model(model_dir, model_type):
    hidden_layers = [1024, 512, 256]
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
            dnn_hidden_units=hidden_layers,
            dnn_dropout=dropout)

    print('Hidden units in each layer:%s' % hidden_layers)
    return m


def process_dataset(filename, quantile, output_path):
    global scaler_fitted, transformer_fitted
    print('dealing with %s' % filename)
    df = pd.read_csv(filename, names=CSV_COLUMNS, sep=',',
                     skipinitialspace=True, skiprows=1, engine='python')
    df = df.drop('attack_cat', axis=1)
    labels = df['label'].astype(int)

    numeric = df[continuous_names + discrete_names].as_matrix()
    symbolic = df[symbolic_names].as_matrix()

    if scaler_fitted is False:
        scaler.fit(numeric)
        scaler_fitted = True

    normalized = scaler.transform(numeric)
    full_columns = symbolic_names + continuous_names + discrete_names
    combined = np.concatenate((symbolic, normalized), axis=1)

    if quantile:
        if transformer_fitted is False:
            transformer.fit(numeric)
            transformer_fitted = True

        augment = transformer.transform(numeric)
        combined = np.concatenate((combined, augment), axis=1)
        full_columns += quantile_names

    combined_df = pd.DataFrame(combined, columns=full_columns,
                               index=labels.index.tolist())
    temp = pd.concat([combined_df, labels], axis=1)
    temp.to_csv(output_path, index=False)

    return full_columns + ['label']


def input_builder(filename, full_columns):
    print('dealing with %s' % filename)
    types = dict()
    for name in discrete_names:
        types[name] = np.int32

    df = pd.read_csv(filename, names=full_columns, sep=',',
                     skipinitialspace=True, skiprows=1,
                     engine='python', dtype=types)
    labels = df['label'].astype(int)
    dataset = df.drop('label', axis=1)
    print('Raw dataset shape:', dataset.shape)
    print('Raw label shape:', labels.shape)

    return tf.estimator.inputs.pandas_input_fn(
        x=dataset, y=labels, batch_size=128, shuffle=True, num_threads=1)


def train_and_eval(model_dir, model_type, train_path, test_path, columns):
    m = build_model(model_dir, model_type)
    train_ib = input_builder(train_path, columns)
    test_ib = input_builder(test_path, columns)
    history = {'train': [], 'test': []}
    for i in range(num_epochs):
        m.train(input_fn=train_ib)

        result = m.evaluate(train_ib)
        history['train'].append(result)
        logger.info('******   Train performance   ******')
        for key in result:
            logger.info("%s: %s" % (key, result[key]))

        result = m.evaluate(test_ib)
        history['test'].append(result)
        logger.info('******   Test performance   ******')
        for key in result:
            logger.info("%s: %s" % (key, result[key]))

    return history


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
train_filename = 'UNSW/UNSW_NB15_training-set.csv'
test_filename = 'UNSW/UNSW_NB15_testing-set.csv'
feature_filename = 'UNSW/feature_names_train_test.csv'
CSV_COLUMNS, symbolic_names, continuous_names, discrete_names = \
    get_feature_names(feature_filename)
upper, lower, small_ranges = discovery_discrete_range(
    [train_filename, test_filename], discrete_names, CSV_COLUMNS)

quantile_names = []
"""
for name in continuous_names + discrete_names:
    quantile_names.append(name + '_quantile')
"""

print(symbolic_names, len(symbolic_names))
print(continuous_names, len(continuous_names))
print(discrete_names, len(discrete_names))
# print(quantile_names, len(quantile_names))

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
"""
for name in quantile_names:
    column = tf.feature_column.numeric_column(name)
    continuous_columns[name] = column
"""

# convert discrete features into categorical columns
discrete_columns = dict()
for name in small_ranges:
    column = tf.feature_column.categorical_column_with_identity(
        name, num_buckets=upper[name] - lower[name] + 1)
    discrete_columns[name] = column

# Build components for the wide model
base_columns = symbolic_columns.values() + discrete_columns.values()
cross_columns = [
    tf.feature_column.crossed_column(
        ['proto', 'service'], hash_bucket_size=1600),
    tf.feature_column.crossed_column(
        ['proto', 'state'], hash_bucket_size=1200),
    tf.feature_column.crossed_column(
        ['service', 'state'], hash_bucket_size=200),
    tf.feature_column.crossed_column(
        ['proto', 'service', 'state'], hash_bucket_size=8000)
]
wide_columns = base_columns + cross_columns
print('#wide components:', len(wide_columns))

# Build components for the deep model
indicator_columns = []  # low dimension categorical features
for name in ['state', 'service']:
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
    # print(name, '|V| =', volcabulary_size)
    dim = np.ceil(np.log2(volcabulary_size))
    print('Embedding size of %s is %d' % (name, dim))
    embedding = tf.feature_column.embedding_column(column, dim)
    embedding_columns.append(embedding)

for (name, column) in discrete_columns.items():
    volcabulary_size = upper[name] - lower[name] + 1
    # print(name, '|V| =', volcabulary_size)
    dim = np.ceil(np.log2(volcabulary_size))
    print('Embedding size of %s is %d' % (name, dim))
    embedding = tf.feature_column.embedding_column(column, int(dim))
    embedding_columns.append(embedding)

print('embedding columns', len(embedding_columns))
deep_columns = indicator_columns + embedding_columns \
    + continuous_columns.values()
print('#deep components:', len(deep_columns))

model_dir = 'WideDeepModel/UNSW/'
num_epochs = 160
batch_size = 40
dropout = 0.2

transformer = QuantileTransformer()
scaler = MinMaxScaler()
scaler_fitted = False
transformer_fitted = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('WD')
hdlr = logging.FileHandler(model_dir + 'Runs%d.accu' % num_epochs)
formatter = logging.Formatter('%(asctime)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

train_path = model_dir + 'aug_train.csv'
test_path = model_dir + 'aug_test.csv'

columns = process_dataset(train_filename, False, train_path)
process_dataset(test_filename, False, test_path)
hist = train_and_eval(model_dir, 'WnD', train_path, test_path, columns)
output = open(model_dir + 'Runs%d.pkl' % (num_epochs), 'wb')
pickle.dump(hist, output)
output.close()

"""
model_dir = 'WideDeepModel/UNSW/'
epoch_list = [160, 240]
num_epochs = np.sum(epoch_list)
hist = dict()
for e in epoch_list:
    output = open(model_dir + 'Runs%d.pkl' % e, 'rb')
    temp = pickle.load(output)
    for (key, value) in temp.items():
        if key not in hist:
            hist[key] = []

        hist[key] += temp[key]
"""
fig, ax1 = plt.subplots()
ax1.plot([x['accuracy'] for x in hist['train']], 'r--', label='Train')
ax1.plot([x['accuracy'] for x in hist['test']], 'b:', label='Test')
ax1.grid(color='k', linestyle=':', linewidth=1)
ax1.set_ylabel('Accuracy')
plt.legend()
fig.tight_layout()
plt.savefig(model_dir + 'accu_%d.pdf' % num_epochs, format='pdf')
plt.close()

fig, ax1 = plt.subplots()
ax1.plot([x['loss'] for x in hist['train']], 'r--', label='Train')
ax1.plot([x['loss'] for x in hist['test']], 'b:', label='Test')
ax1.grid(color='k', linestyle=':', linewidth=1)
ax1.set_ylabel('Loss')
plt.legend()
fig.tight_layout()
plt.savefig(model_dir + 'loss_%d.pdf' % num_epochs, format='pdf')
plt.close()
