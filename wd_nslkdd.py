from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import logging
import os
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from preprocess.nslkdd import get_feature_names, get_categorical_values


def build_model(model_dir, model_type):
    hidden_layers = [256, 128, 64]
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
            dnn_hidden_units=hidden_layers, dnn_dropout=dropout)

    print('Hidden units in each layer:%s' % hidden_layers)
    return m


def augment_dataset(fname, output_path):
    global scaler_fitted, transformer_fitted
    df_data = pd.read_csv(tf.gfile.Open(fname), names=CSV_COLUMNS, sep=',',
                          skipinitialspace=True, engine='python', skiprows=1)
    data = df_data.drop('difficulty', axis=1)
    labels = df_data['traffic'].apply(lambda x: x != 'normal').astype(int)
    data = data.drop('traffic', axis=1)
    print('Raw dataset shape', data.shape)
    print('Raw label shape', labels.shape)

    numeric = data[continuous_names + discrete_names].as_matrix()

    if scaler_fitted:
        print('Scaler already fitted')
        numeric = scaler.transform(numeric)
    else:
        print('First time fit scaler')
        scaler.fit(numeric)
        numeric = scaler.transform(numeric)
        scaler_fitted = True

    if transformer_fitted:
        print('Transformer already fitted')
        augment = transformer.transform(numeric)
    else:
        print('First time fit transformer')
        transformer.fit(numeric)
        augment = transformer.transform(numeric)
        transformer_fitted = True

    temp = np.concatenate((data.as_matrix(), augment), axis=1)
    columns = CSV_COLUMNS[:-2] + quantile_names
    combined = pd.DataFrame(temp, labels.index.tolist(), columns)
    save = pd.concat([combined, labels], axis=1)
    save.to_csv(output_path, index=False)

    return columns + ['label']


def input_builder(data_file, columns):
    df_data = pd.read_csv(tf.gfile.Open(data_file), names=columns,
                          sep=',', skipinitialspace=True,
                          engine='python', skiprows=1)
    labels = df_data['label'].astype(int)
    dataset = df_data.drop('label', axis=1)
    dataset['land'] = dataset['land'].astype(str)
    dataset['login'] = dataset['login'].astype(str)
    dataset['guest_login'] = dataset['guest_login'].astype(str)
    dataset['host_login'] = dataset['host_login'].astype(str)
    # print('Dataset shape', dataset.shape)
    # print('Label shape', labels.shape)
    ib = tf.estimator.inputs.pandas_input_fn(dataset, labels, batch_size,
                                             shuffle=True, num_threads=1)
    return ib


def train_and_eval(model_dir, mtype, columns, train_filename, test_filename):
    m = build_model(model_dir, mtype)
    train_ib = input_builder(train_filename, columns)
    test_ib = input_builder(test_filename, columns)
    history = {'train': [], 'test': []}
    for i in range(num_epochs):
        m.train(input_fn=train_ib)
        results = m.evaluate(train_ib)
        history['train'].append(results)
        logger.info('******   Train performance   ******')
        for key in results:
            logger.info("%s: %s" % (key, results[key]))

        results = m.evaluate(input_fn=test_ib)
        history['test'].append(results)
        logger.info('******   Test performance   ******')
        for key in results:
            logger.info("%s: %s" % (key, results[key]))

    return history


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CSV_COLUMNS, symbolic_names, continuous_names, discrete_names = \
    get_feature_names('NSLKDD/feature_names.csv')
header = ""
for name in CSV_COLUMNS:
    header += name + ','

print('Feature names: ', header)
print(symbolic_names)
print(continuous_names)
print(discrete_names)

quantile_names = []
for name in continuous_names + discrete_names:
    quantile_names.append(name + '_quantile')

# Build wide columns
protocol = tf.feature_column.categorical_column_with_vocabulary_list(
    'protocol', get_categorical_values('protocol'))
service = tf.feature_column.categorical_column_with_vocabulary_list(
    'service', get_categorical_values('service'))
flag = tf.feature_column.categorical_column_with_vocabulary_list(
    'flag', get_categorical_values('flag'))
land = tf.feature_column.categorical_column_with_vocabulary_list(
    'land', get_categorical_values('land'))
login = tf.feature_column.categorical_column_with_vocabulary_list(
    'login', get_categorical_values('login'))
host_login = tf.feature_column.categorical_column_with_vocabulary_list(
    'host_login', get_categorical_values('host_login'))
guest_login = tf.feature_column.categorical_column_with_vocabulary_list(
    'guest_login', get_categorical_values('guest_login'))

base_columns = [protocol, service, flag, land,
                login, host_login, guest_login]
cross_columns = [
    tf.feature_column.crossed_column(
        ['protocol', 'service'], hash_bucket_size=240),
    tf.feature_column.crossed_column(
        ['service', 'flag'], hash_bucket_size=800),
    tf.feature_column.crossed_column(
        ['land', 'login', 'host_login', 'guest_login'], hash_bucket_size=8),
    tf.feature_column.crossed_column(
        ['service', 'login', 'host_login'], hash_bucket_size=290),
    tf.feature_column.crossed_column(
        ['protocol', 'host_login', 'guest_login'], hash_bucket_size=12),
    tf.feature_column.crossed_column(
        ['protocol', 'service', 'flag'], hash_bucket_size=2400),
]
wide_columns = base_columns + cross_columns
print("size of wide columns", len(wide_columns))

indicator_columns = [
    tf.feature_column.indicator_column(protocol),
    tf.feature_column.indicator_column(land),
    tf.feature_column.indicator_column(login),
    tf.feature_column.indicator_column(host_login),
    tf.feature_column.indicator_column(guest_login),
]
embedding_columns = [
    tf.feature_column.embedding_column(service, dimension=16),
    tf.feature_column.embedding_column(flag, dimension=8),
]
# Build continous columns
continuous_columns = []
for name in continuous_names + discrete_names + quantile_names:
    column = tf.feature_column.numeric_column(name)
    continuous_columns.append(column)

deep_columns = indicator_columns + embedding_columns + continuous_columns
print("size of deep columns", len(deep_columns))

train_filename = 'NSLKDD/KDDTrain.csv'
test_filename = 'NSLKDD/KDDTest.csv'
model_dir = 'WideDeepModel/NSLKDD/'
train_path = model_dir + 'aug_train.csv'
test_path = model_dir + 'aug_test.csv'
num_epochs = 160
batch_size = 160
dropout = 0.2

scaler = MinMaxScaler()
transformer = QuantileTransformer()
scaler_fitted = False
transformer_fitted = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ModalityNets')
hdlr = logging.FileHandler(model_dir + 'Runs%d.accu' % num_epochs)
formatter = logging.Formatter('%(asctime)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

columns = augment_dataset(train_filename, train_path)
augment_dataset(test_filename, model_dir + 'aug_test.csv')
hist = train_and_eval(model_dir, 'WnD', columns, train_path, test_path)
output = open(model_dir + 'Runs%d.pkl' % (num_epochs), 'wb')
pickle.dump(hist, output)
output.close()

"""
model_dir = 'WideDeepModel/NSLKDD/'
num_epochs = 320
output = open(model_dir + 'Runs%d.pkl' % (num_epochs), 'rb')
hist = pickle.load(output)
"""
fig, ax1 = plt.subplots()
ax1.plot([x['accuracy'] for x in hist['train']], 'r--')
ax1.set_ylabel('train', color='r')
ax1.tick_params('y', colors='r')
ax2 = ax1.twinx()
ax2.plot([x['accuracy'] for x in hist['test']], 'b:')
ax2.set_ylabel('test', color='b')
ax2.tick_params('y', colors='b')
ax1.grid(color='k', linestyle=':', linewidth=1)
ax2.grid(color='k', linestyle=':', linewidth=1)
fig.tight_layout()
plt.savefig(model_dir + 'accu_%d.pdf' % num_epochs, format='pdf')
plt.close()

fig, ax1 = plt.subplots()
ax1.plot([x['loss'] for x in hist['train']], 'r--')
ax1.set_ylabel('train', color='r')
ax1.tick_params('y', colors='r')
ax2 = ax1.twinx()
ax2.plot([x['loss'] for x in hist['test']], 'b:')
ax2.set_ylabel('test', color='b')
ax2.tick_params('y', colors='b')
ax1.grid(color='k', linestyle=':', linewidth=1)
ax2.grid(color='k', linestyle=':', linewidth=1)
fig.tight_layout()
plt.grid(which='both', color='k', linestyle=':', linewidth=1)
plt.savefig(model_dir + 'loss_%d.pdf' % num_epochs, format='pdf')
plt.close()
