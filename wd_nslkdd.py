from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import logging
from math import ceil
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from preprocess.nslkdd import get_feature_names, get_categorical_values
from preprocess.nslkdd import attack_category_map
from netlearner.utils import measure_prediction


def plot_history(train_loss, valid_loss, test_loss, fig_dir):
    fig, ax1 = plt.subplots()
    ln1 = ax1.plot(train_loss, 'r--', label='Train')
    if len(valid_loss) == 0:
        valid_loss = [0.0] * len(train_loss)

    ln2 = ax1.plot(valid_loss, 'b:', label='Valid')
    ax1.set_ylabel('Train/Valid Loss', color='r')

    ax2 = ax1.twinx()
    ln3 = ax2.plot(test_loss, 'g-.', label='Test')
    ax2.set_ylabel('Test Loss', color='g')

    lns = ln1 + ln2 + ln3
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels, loc='upper left')

    ax1.grid(color='k', linestyle=':', linewidth=1)
    ax2.grid(color='k', linestyle=':', linewidth=1)
    fig.tight_layout()
    plt.savefig(fig_dir + 'history.pdf', format='pdf')
    plt.close()


def build_model(model_dir, model_type):
    hidden_layers = [800, 480]
    label_names = label_mapping.keys()
    m = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_layers,
        label_vocabulary=label_names,
        dnn_dropout=dropout,
        n_classes=len(label_names))
    print('Hidden units in each layer:%s' % hidden_layers)
    return m


def process_dataset(fname, output_path):
    global scaler_fitted, transformer_fitted
    df_data = pd.read_csv(tf.gfile.Open(fname), names=CSV_COLUMNS, sep=',',
                          skipinitialspace=True, engine='python', skiprows=1)
    data = df_data.drop('difficulty', axis=1)
    labels = df_data['traffic'].apply(lambda x: attack_category_map[x])
    data = data.drop('traffic', axis=1)
    print('Raw dataset shape', data.shape)
    print('Raw label shape', labels.shape)

    numeric = data[continuous_names + discrete_names].as_matrix()
    symbolic = data[symbolic_names].as_matrix()
    if scaler_fitted is False:
        print('First time fit scaler')
        scaler.fit(numeric)
        scaler_fitted = True

    normalized = scaler.transform(numeric)
    full_columns = symbolic_names + continuous_names + discrete_names
    combined = np.concatenate((symbolic, normalized), axis=1)

    if len(quantile_names) > 0:
        if transformer_fitted is False:
            transformer.fit(numeric)
            transformer_fitted = True

        augment = transformer.transform(numeric)
        combined = np.concatenate((combined, augment), axis=1)
        full_columns += quantile_names

    combined_df = pd.DataFrame(combined, columns=full_columns,
                               index=labels.index.tolist())
    save = pd.concat([combined_df, labels], axis=1)
    save.to_csv(output_path, index=False)
    return full_columns + ['label']


def input_builder(data_file, columns):
    df_data = pd.read_csv(tf.gfile.Open(data_file), names=columns,
                          sep=',', skipinitialspace=True,
                          engine='python', skiprows=1)
    labels = df_data['label']
    labels_ohe = np.zeros((labels.shape[0], len(label_mapping)))
    for (i, x) in enumerate(labels):
        labels_ohe[i][label_mapping[x]] = 1.0

    dataset = df_data.drop('label', axis=1)
    dataset['land'] = dataset['land'].astype(str)
    dataset['login'] = dataset['login'].astype(str)
    dataset['guest_login'] = dataset['guest_login'].astype(str)
    dataset['host_login'] = dataset['host_login'].astype(str)
    print('Dataset shape', dataset.shape)
    print('Label shape', labels.shape)
    ib = tf.estimator.inputs.pandas_input_fn(dataset, labels, batch_size,
                                             shuffle=True, num_threads=1)
    return ib, labels_ohe


def train_and_eval(model_dir, mtype, columns, train_filename, test_filename):
    m = build_model(model_dir, mtype)
    test_ib, ohe = input_builder(test_filename, columns)
    history = {'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': []}
    num_samples = ohe.shape[0]
    for i in range(num_epochs):
        train_ib, _ = input_builder(train_filename, columns)
        m.train(input_fn=train_ib, steps=ceil(num_samples / batch_size))
        results = m.evaluate(train_ib)
        history['train_loss'].append(results['average_loss'])
        history['train_acc'].append(results['accuracy'])
        logger.info('******   Train performance   ******')
        for key in results:
            logger.info("%s: %s" % (key, results[key]))

        results = m.evaluate(input_fn=test_ib)
        history['test_loss'].append(results['average_loss'])
        history['test_acc'].append(results['accuracy'])
        logger.info('******   Test performance   ******')
        for key in results:
            logger.info("%s: %s" % (key, results[key]))

    opt_epochs = np.argmin(history['test_acc'])
    opt_accu = np.max(history['test_acc'])
    print('Test accuracy = %s at epoch %d' % (opt_accu, opt_epochs))

    predictions = np.zeros_like(ohe)
    cnt = 0
    for (i, x) in enumerate(list(m.predict(test_ib))):
        cnt += 1
        predictions[i][x['class_ids'][0]] = 1.0

    print(cnt)
    conf_table = measure_prediction(np.array(predictions), ohe, model_dir)
    history['confusion_table'] = conf_table
    print(conf_table)
    plot_history(history['train_loss'], [], history['test_loss'], model_dir)
    return history


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
CSV_COLUMNS, symbolic_names, continuous_names, discrete_names = \
    get_feature_names('NSLKDD/feature_names.csv')
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
        ['protocol', 'service'], hash_bucket_size=200),
    tf.feature_column.crossed_column(
        ['service', 'flag'], hash_bucket_size=480),
    tf.feature_column.crossed_column(
        ['land', 'login', 'host_login', 'guest_login'], hash_bucket_size=8),
    tf.feature_column.crossed_column(
        ['service', 'login', 'host_login'], hash_bucket_size=290),
    tf.feature_column.crossed_column(
        ['protocol', 'host_login', 'guest_login'], hash_bucket_size=12),
    tf.feature_column.crossed_column(
        ['protocol', 'service', 'flag'], hash_bucket_size=1000),
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
num_epochs = 2
batch_size = 64
dropout = 0.2
label_mapping = {'normal': 0, 'probe': 1, 'dos': 2, 'u2r': 3, 'r2l': 4}
class_weights = {'normal': 0.15, 'probe': 0.2,
                 'dos': 0.15, 'u2r': 0.3, 'r2l': 0.2}
transformer = QuantileTransformer()
transformer_fitted = False
scaler = MinMaxScaler()
scaler_fitted = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('WD-NSLKDD')
hdlr = logging.FileHandler(model_dir + 'Runs%d.accu' % num_epochs)
formatter = logging.Formatter('%(asctime)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

columns = process_dataset(train_filename, train_path)
process_dataset(test_filename, test_path)
hist = train_and_eval(model_dir, 'WnD', columns, train_path, test_path)
output = open(model_dir + 'Runs%d.pkl' % (num_epochs), 'wb')
pickle.dump(hist, output)
output.close()

"""
model_dir = 'WideDeepModel/NSLKDD/'
epoch_list = [1]
num_epochs = sum(epoch_list)
hist = {'train': [], 'test': []}
for e in epoch_list:
    output = open(model_dir + 'Runs%d.pkl' % e, 'rb')
    temp = pickle.load(output)
    print(temp)
    hist['train'] += temp['train']
    hist['test'] += temp['test']
"""
