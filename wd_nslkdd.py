from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf
from preprocess.nslkdd import get_feature_names, get_categorical_values

CSV_COLUMNS, symbolic_names, continuous_names = get_feature_names()
header = ""
for name in CSV_COLUMNS:
    header += name + ','

print('Feature names: ', header)

# Build symbolic columns
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
symbolic_columns = [protocol, service, flag, land,
                    login, host_login, guest_login]
# Build continous columns
continuous_columns = []
for name in continuous_names:
    column = tf.feature_column.numeric_column(name)
    continuous_columns.append(column)

# Build wide columns
base_columns = [protocol, service, flag, land,
                login, host_login, guest_login]
cross_columns = [
    tf.feature_column.crossed_column(
        ['protocol', 'service'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['service', 'flag'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['land', 'login', 'host_login', 'guest_login'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['service', 'login', 'host_login'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ['protocol', 'host_login', 'guest_login'], hash_bucket_size=1000),
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
    tf.feature_column.embedding_column(service, dimension=5),
    tf.feature_column.embedding_column(flag, dimension=2),
]
deep_columns = continuous_columns + indicator_columns + embedding_columns
print("size of deep columns", len(deep_columns))


def build_model(model_dir, model_type):
    if model_type == 'wide':
        m = tf.estimator.LinearClassifier(
            model_dir=model_dir, feature_columns=wide_columns)
    elif model_type == 'deep':
        m = tf.estimator.DNNClassifier(
            model_dir=model_dir, feature_columns=deep_columns,
            hidden_units=[400, 200, 50])
    else:
        m = tf.estimator.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[400, 200, 50])

    return m


def input_builder(data_file, num_epochs, shuffle):
    df_data = pd.read_csv(tf.gfile.Open(data_file),
                          names=CSV_COLUMNS,
                          sep=',',
                          skipinitialspace=True,
                          engine='python',
                          skiprows=1)
    data = df_data.drop('difficulty', axis=1)
    data = data.drop('traffic', axis=1)

    data['land'] = data['land'].astype(str)
    data['login'] = data['login'].astype(str)
    data['guest_login'] = data['guest_login'].astype(str)
    data['host_login'] = data['host_login'].astype(str)
    # data = df_data.dropna(how='any', axis=0)
    labels = df_data['traffic'].apply(lambda x: x != 'normal').astype(int)

    print('Raw dataset shape', data.shape)
    print('Raw label shape', labels.shape)
    return tf.estimator.inputs.pandas_input_fn(
        x=data, y=labels, batch_size=40, num_epochs=num_epochs,
        shuffle=shuffle, num_threads=4)


def train_and_eval(model_dir, model_type, train_steps,
                   train_filename, test_filename):
    m = build_model(model_dir, model_type)
    m.train(
        input_fn=input_builder(train_filename, num_epochs=None, shuffle=True),
        steps=train_steps)

    results = m.evaluate(
        input_fn=input_builder(test_filename, num_epochs=1, shuffle=False),
        steps=None)
    for key in results:
        print("%s: %s" % (key, results[key]))


train_filename = 'NSLKDD/KDDTrain+.txt'
test_filename = 'NSLKDD/KDDTest+.txt'
model_dir = 'WideDeepModel'
train_steps = 8000
train_and_eval(model_dir, 'wide+deep', train_steps,
               train_filename, test_filename)
