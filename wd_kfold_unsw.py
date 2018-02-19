from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
from sklearn.model_selection import StratifiedKFold
from preprocess.unsw import get_feature_names, discovery_feature_volcabulary
from preprocess.unsw import generate_header, discovery_discrete_range
from netlearner.utils import measure_prediction
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os


def test_splitted_data_builder(train_path, test_path):
    for i in range(fold):
        train_fold = train_path + '.train_fold%d.csv' % i
        valid_fold = train_path + '.valid_fold%d.csv' % i
        train_ib, _ = input_builder(train_fold, columns)
        valid_ib, _ = input_builder(valid_fold, columns)

    test_ib, ohe = input_builder(test_path, columns)
    train_ib, _ = input_builder(train_path, columns)


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
    label_names = label_map.keys()
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


def process_dataset(filename, output_path, split):
    global scaler_fitted, transformer_fitted
    print('Process %s' % filename)
    df = pd.read_csv(filename, names=CSV_COLUMNS, sep=',',
                     skipinitialspace=True, skiprows=1, engine='python')
    labels = df['label'] if binary_class else df['attack_cat']
    df.drop(['label', 'attack_cat'], axis=1)

    numeric = df[continuous_names + discrete_names].as_matrix()
    symbolic = df[symbolic_names].as_matrix()

    if scaler_fitted is False:
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

    print('Raw dataset shape', combined.shape)
    print('Raw label shape', labels.shape)
    if split is True:
        skf = StratifiedKFold(n_splits=fold)
        X = combined.copy()
        y = np.reshape(labels, (-1, 1))
        i = 0
        for train_index, valid_index in skf.split(X, labels):
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


def input_builder(filename, full_columns):
    print('Building input for %s' % filename)
    dtypes = dict(zip(discrete_names, [np.int32] * len(discrete_names)))
    df = pd.read_csv(filename, names=full_columns, sep=',',
                     skipinitialspace=True, skiprows=1,
                     engine='python', dtype=dtypes)
    labels = df['label'].astype(str)
    labels_ohe = np.zeros((labels.shape[0], len(label_map)))
    for (i, x) in enumerate(labels):
        labels_ohe[i][label_map[x]] = 1.0

    dataset = df.drop('label', axis=1)
    print('Dataset shape:', dataset.shape)
    print('Label shape:', labels.shape)
    ib = tf.estimator.inputs.pandas_input_fn(dataset, labels, batch_size,
                                             shuffle=True, num_threads=1)
    return ib, labels_ohe


def train_and_eval(model_dir, train_path, test_path, columns):
    hist = {'train_loss': [], 'valid_loss': [],
            'test_loss': [], 'test_acc': []}
    fold_train_loss, fold_valid_loss = [], []
    for i in range(fold):
        train_fold = train_path + '.train_fold%d.csv' % i
        valid_fold = train_path + '.valid_fold%d.csv' % i
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
        print('Train fold %d finished' % i)

    hist['train_loss'] = np.mean(fold_train_loss, axis=0)
    hist['valid_loss'] = np.mean(fold_valid_loss, axis=0)
    opt_epochs = np.argmin(hist['valid_loss'])
    print('Optimal #Epochs:', opt_epochs + 1)
    hist['opt_epochs'] = opt_epochs + 1

    test_ib, ohe = input_builder(test_path, columns)
    m = build_model(None)
    for i in range(num_epochs):
        train_ib, _ = input_builder(train_path, columns)
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


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
train_filename = 'UNSW/UNSW_NB15_training-set.csv'
test_filename = 'UNSW/UNSW_NB15_testing-set.csv'
feature_filename = 'UNSW/feature_names_train_test.csv'
CSV_COLUMNS, symbolic_names, continuous_names, discrete_names = \
    get_feature_names(feature_filename)
upper, lower, small_ranges = discovery_discrete_range(
    [train_filename, test_filename], discrete_names, CSV_COLUMNS)

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
for name in continuous_names + discrete_names + quantile_names:
    column = tf.feature_column.numeric_column(name)
    continuous_columns[name] = column

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
        ['proto', 'service'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['proto', 'state'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['service', 'state'], hash_bucket_size=200),
    tf.feature_column.crossed_column(
        ['proto', 'service', 'state'], hash_bucket_size=1600)
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
train_path = model_dir + 'aug_train.csv'
test_path = model_dir + 'aug_test.csv'
num_epochs = 400
batch_size = 64
dropout = 0.2
fold = 5
transformer = QuantileTransformer()
transformer_fitted = False
scaler = MinMaxScaler()
scaler_fitted = False
label_map = {'0': 0, '1': 1}
# label_map = {"Normal": 0, "Backdoor": 1, "Analysis": 2, "Fuzzers": 3,
# "Reconnaissance": 4, "Exploits": 5, "DoS": 6,
# "Shellcode": 7, "Worms": 8, "Generic": 9}
binary_class = (len(label_map) == 2)
columns = process_dataset(train_filename, train_path, split=True)
process_dataset(test_filename, test_path, split=False)
hist = train_and_eval(model_dir, train_path, test_path, columns)
plot_history(hist['train_loss'], hist['valid_loss'],
             hist['test_loss'], model_dir)
output = open(model_dir + 'Runs%d.pkl' % (num_epochs), 'wb')
pickle.dump(hist, output)
output.close()
