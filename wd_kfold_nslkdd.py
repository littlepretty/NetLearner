from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from preprocess.nslkdd import get_feature_names, get_categorical_values
from preprocess.nslkdd import attack_category_map
from netlearner.utils import measure_prediction


def plot_history(train_loss, valid_loss, test_loss, fig_dir):
    fig, ax1 = plt.subplots()
    ln1 = ax1.plot(train_loss, 'r--', label='Train')
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


def build_model(model_dir):
    hidden_layers = [800, 480]
    label_names = label_mapping.keys()
    m = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_layers,
        dnn_dropout=dropout,
        label_vocabulary=label_names,
        n_classes=len(label_names))
    print('Hidden units in each layer:%s' % hidden_layers)
    return m


def process_dataset(fname, output_path, split=False):
    global scaler_fitted, transformer_fitted
    print('Process %s' % fname)
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

    if split is True:
        skf = StratifiedKFold(n_splits=fold)
        X = combined.copy()
        y = np.reshape(labels, (-1, 1))
        i = 0
        for train_index, valid_index in skf.split(X, y):
            train_dataset, valid_dataset = X[train_index], X[valid_index]
            train_labels, valid_labels = y[train_index], y[valid_index]

            train = np.concatenate((train_dataset, train_labels), axis=1)
            train_df = pd.DataFrame(train, columns=full_columns + ['label'])
            train_df.to_csv(output_path + '.train_fold%d.csv' % i, index=False)

            valid = np.concatenate((valid_dataset, valid_labels), axis=1)
            valid_df = pd.DataFrame(valid, columns=full_columns + ['label'])
            valid_df.to_csv(output_path + '.valid_fold%d.csv' % i, index=False)
            i += 1

    combined_df = pd.DataFrame(combined, columns=full_columns,
                               index=labels.index.tolist())
    save = pd.concat([combined_df, labels], axis=1)
    save.to_csv(output_path, index=False)
    return full_columns + ['label']


def input_builder(data_file, columns):
    print('Building input for %s' % data_file)
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


def train_and_eval(model_dir, columns, train_filename, test_filename):
    hist = {'train_loss': [], 'valid_loss': [],
            'test_loss': [], 'test_acc': []}
    fold_train_loss, fold_valid_loss = [], []
    for i in range(fold):
        train_fold = train_filename + '.train_fold%d.csv' % i
        valid_fold = train_filename + '.valid_fold%d.csv' % i
        m = build_model(model_dir=None)
        train_loss, valid_loss = [], []
        for _ in range(num_epochs):
            train_ib, _ = input_builder(train_fold, columns)
            valid_ib, _ = input_builder(valid_fold, columns)
            m.train(train_ib)
            scores = m.evaluate(input_fn=train_ib)
            train_loss.append(scores['average_loss'])
            scores = m.evaluate(input_fn=valid_ib)
            valid_loss.append(scores['average_loss'])

        fold_train_loss.append(train_loss)
        fold_valid_loss.append(valid_loss)

    hist['train_loss'] = np.mean(fold_train_loss, axis=0)
    hist['valid_loss'] = np.mean(fold_valid_loss, axis=0)
    opt_epochs = np.argmin(hist['valid_loss'])
    print('Optimal #Epochs:', opt_epochs + 1)
    hist['opt_epochs'] = opt_epochs + 1

    test_ib, ohe = input_builder(test_filename, columns)
    m = build_model(None)
    for i in range(num_epochs):
        train_ib, _ = input_builder(train_filename, columns)
        m.train(input_fn=train_ib)
        scores = m.evaluate(test_ib)
        hist['test_loss'].append(scores['average_loss'])
        hist['test_acc'].append(scores['accuracy'])

    hist['test_acc_report'] = hist['test_acc'][opt_epochs]
    print('Test accuracy = %s' % hist['test_acc_report'])

    predictions = np.zeros_like(ohe)
    for (i, x) in enumerate(list(m.predict(test_ib))):
        predictions[i][x['class_ids'][0]] = 1.0

    conf_table = measure_prediction(np.array(predictions), ohe, model_dir)
    hist['confusion_table'] = conf_table
    print(conf_table)
    return hist


os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
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
fold = 5
num_epochs = 360
batch_size = 64
dropout = 0.2
label_mapping = {'normal': 0, 'probe': 1, 'dos': 2, 'u2r': 3, 'r2l': 4}
class_weights = {'normal': 0.15, 'probe': 0.2,
                 'dos': 0.15, 'u2r': 0.3, 'r2l': 0.2}
transformer = QuantileTransformer()
transformer_fitted = False
scaler = MinMaxScaler()
scaler_fitted = False

columns = process_dataset(train_filename, train_path, split=True)
process_dataset(test_filename, test_path, split=False)
hist = train_and_eval(model_dir, columns, train_path, test_path)
plot_history(hist['train_loss'], hist['valid_loss'],
             hist['test_loss'], model_dir)
output = open(model_dir + 'Runs%d.pkl' % (num_epochs), 'wb')
pickle.dump(hist, output)
output.close()
