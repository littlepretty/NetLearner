from keras.models import Model
from keras.layers import Dense, Input, concatenate, Flatten, Dropout
# from keras import regularizers
from keras.layers import Embedding, BatchNormalization
from keras.callbacks import CSVLogger

from preprocess import unsw, nslkdd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pprint import pprint
import pickle
import pandas as pd
import numpy as np
import logging


def get_dataset(dataset_filename, headers, dataset_name):
    df = pd.read_csv(dataset_filename, names=headers, sep=',',
                     skipinitialspace=True, skiprows=1,
                     engine='python')
    num_classes = 2
    if dataset_name == 'unsw':
        X = df.drop('attack_cat', axis=1)
        labels = df['label'].astype(int).as_matrix()
        y = np.zeros(shape=(labels.shape[0], num_classes))
        for (i, l) in enumerate(labels):
            y[i, l] = 1
        return X, y
    elif dataset_name == 'nsl':
        logger.debug(headers)
        X = df.drop('difficulty', axis=1)
        traffic = df['traffic'].as_matrix()
        y = np.zeros(shape=(traffic.shape[0], num_classes))
        for (i, label) in enumerate(traffic):
            if label == 'normal':
                y[i, 0] = 1
            else:
                y[i, 1] = 1

        return X, y


def build_embeddings(symbolic_features, integer_features,
                     embeddings, large_discrete, merged_inputs,
                     X, test_X, train_dict, test_dict, dataset):
    """Define embedding layers/inputs"""
    merged_dim = 0
    for (name, values) in symbolic_features.items():
        feature_name = name + '_' + dataset
        column = Input(shape=(1, ), name=feature_name)
        merged_inputs.append(column)
        raw_data = X[name].as_matrix()
        test_raw_data = test_X[name].as_matrix()
        le = LabelEncoder()
        le.fit(np.concatenate((raw_data, test_raw_data), axis=0))
        train_dict[feature_name] = le.transform(raw_data)
        test_dict[feature_name] = le.transform(test_raw_data)

        dim_V = len(values)
        dim_E = int(min(7, np.ceil(np.log2(dim_V))))
        logger.debug('Dimension of %s E=%s and V=%s' % (name, dim_E, dim_V))
        temp = Embedding(output_dim=dim_E, input_dim=dim_V,
                         input_length=1, name='embed_%s' % feature_name)(column)
        temp = Flatten(name='flat_%s' % feature_name)(temp)
        embeddings.append(temp)
        merged_dim += dim_E

    for (name, values) in integer_features.items():
        feature_name = name + '_' + dataset
        raw_data = X[name].astype('int64').as_matrix()
        test_raw_data = test_X[name].astype('int64').as_matrix()
        dim_V = int(values['max'] - values['min'] + 1)
        if dim_V == 1:
            continue

        column = Input(shape=(1, ), name=feature_name)
        merged_inputs.append(column)

        if dim_V < 8096:
            train_dict[feature_name] = raw_data - values['min']
            test_dict[feature_name] = test_raw_data - values['min']
            dim_E = int(min(5, np.ceil(np.log2(dim_V))))
            logger.debug('Dimension of %s E=%s and V=%s' % (name, dim_E, dim_V))
            temp = Embedding(output_dim=dim_E, input_dim=dim_V,
                             input_length=1,
                             name='embed_%s' % feature_name)(column)
            temp = Flatten(name='flat_%s' % feature_name)(temp)
            embeddings.append(temp)
            merged_dim += dim_E
        else:
            large_discrete.append(column)
            logger.debug('[%s] is too large so is treated as continuous'
                         % feature_name)
            mm = MinMaxScaler()
            raw_data = raw_data.reshape((len(raw_data), 1))
            test_raw_data = test_raw_data.reshape((len(test_raw_data), 1))
            mm.fit(np.concatenate((raw_data, test_raw_data), axis=0))
            train_dict[feature_name] = mm.transform(raw_data)
            test_dict[feature_name] = mm.transform(test_raw_data)
            merged_dim += 1

    return merged_dim


def build_continuous(continuous_features, merged_inputs,
                     X, test_X, train_dict, test_dict, dataset):
    continuous_inputs = Input(shape=(len(continuous_features), ),
                              name='continuous_' + dataset)
    merged_inputs.append(continuous_inputs)
    raw_data = X[continuous_features.keys()].as_matrix()
    test_raw_data = test_X[continuous_features.keys()].as_matrix()
    mm = MinMaxScaler()
    mm.fit(np.concatenate((raw_data, test_raw_data), axis=0))
    train_dict['continuous_' + dataset] = mm.transform(raw_data)
    test_dict['continuous_' + dataset] = mm.transform(test_raw_data)

    return continuous_inputs


def get_unsw_data():
    dataset_names = ['UNSW/UNSW_NB15_%s-set.csv' % x
                     for x in ['training', 'testing']]
    feature_file = 'UNSW/feature_names_train_test.csv'

    headers, _, _, _ = unsw.get_feature_names(feature_file)
    symbolic_features = unsw.discovery_feature_volcabulary(dataset_names)
    integer_features = unsw.discovery_integer_map(feature_file, dataset_names)
    continuous_features = unsw.discovery_continuous_map(feature_file,
                                                        dataset_names)
    X, y = get_dataset(dataset_names[0], headers, 'unsw')
    test_X, test_y = get_dataset(dataset_names[1], headers, 'unsw')

    train_dict = dict()
    test_dict = dict()
    merged_inputs = []
    embeddings = []
    large_discrete = []
    merged_dim = 0
    merged_dim += build_embeddings(symbolic_features, integer_features,
                                   embeddings, large_discrete, merged_inputs,
                                   X, test_X, train_dict, test_dict, 'unsw')
    merged_dim += len(continuous_features)
    cont_component = build_continuous(continuous_features,
                                      merged_inputs, X, test_X,
                                      train_dict, test_dict, 'unsw')
    logger.info('merge input_dim for UNSW-NB dataset = %s' % merged_dim)

    merge = concatenate(embeddings + large_discrete + [cont_component],
                        name='concate_features_unsw')

    return merge, merged_inputs, train_dict, test_dict, y, test_y


def get_nsl_data():
    dataset_names = ['NSLKDD/KDD%s.csv' % x for x in ['Train', 'Test']]
    feature_file = 'NSLKDD/feature_names.csv'
    headers, _, _, _ = nslkdd.get_feature_names(feature_file)
    symbolic_features = nslkdd.discovery_feature_volcabulary(dataset_names)
    integer_features = nslkdd.discovery_integer_map(feature_file, dataset_names)
    continuous_features = nslkdd.discovery_continuous_map(feature_file,
                                                          dataset_names)
    X, y = get_dataset(dataset_names[0], headers, 'nsl')
    test_X, test_y = get_dataset(dataset_names[1], headers, 'nsl')

    train_dict = dict()
    test_dict = dict()
    merged_inputs = []
    embeddings = []
    large_discrete = []
    merged_dim = 0
    merged_dim += build_embeddings(symbolic_features, integer_features,
                                   embeddings, large_discrete, merged_inputs,
                                   X, test_X, train_dict, test_dict, 'nsl')
    merged_dim += len(continuous_features)
    cont_component = build_continuous(continuous_features,
                                      merged_inputs, X, test_X,
                                      train_dict, test_dict, 'nsl')
    logger.info('merge input_dim for NSLKDD dataset = %s' % merged_dim)

    merge = concatenate(embeddings + large_discrete + [cont_component],
                        name='concate_features_nsl')

    return merge, merged_inputs, train_dict, test_dict, y, test_y


def shared_models(unsw, nsl, unsw_inputs, nsl_inputs, hidden, dropprob=0.2):
    h1_unsw = Dense(hidden[0], activation='relu', name='h1_unsw')(unsw)
    h1_nsl = Dense(hidden[0], activation='relu', name='h1_nsl')(nsl)
    h1_unsw = Dropout(dropprob)(h1_unsw)
    h1_nsl = Dropout(dropprob)(h1_nsl)

    shared_h2 = Dense(hidden[1], activation='relu', name='shared_layer')
    h2_unsw = shared_h2(h1_unsw)
    h2_nsl = shared_h2(h1_nsl)

    bn_unsw = BatchNormalization(name='bn_unsw')(h2_unsw)
    bn_nsl = BatchNormalization(name='bn_nsl')(h2_nsl)

    shared_h3 = Dense(hidden[2], activation='sigmoid', name='h3_unsw')
    h3_unsw = shared_h3(bn_unsw)
    h3_nsl = shared_h3(bn_nsl)

    shared_sm = Dense(2, activation='softmax', name='output')
    output_unsw = shared_sm(h3_unsw)
    output_nsl = shared_sm(h3_nsl)

    unsw_model = Model(inputs=unsw_inputs, outputs=output_unsw)
    unsw_model.compile(optimizer='adam', loss='binary_crossentropy',
                       metrics=['accuracy'])

    nsl_model = Model(inputs=nsl_inputs, output=output_nsl)
    nsl_model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])

    unsw_model.summary()
    nsl_model.summary()
    return unsw_model, nsl_model


def run_main(hidden):
    num_epochs = 20
    batch_size = 100
    m1, m2 = shared_models(unsw_tens, nsl_tens, unsw_inputs, nsl_inputs, hidden)
    csv_logger = CSVLogger(root + 'history.log', append=True)
    unsw_loss, nsl_loss = [], []
    for _ in range(num_epochs):
        m1.fit(X_unsw, y_unsw, batch_size, epochs=1, callbacks=[csv_logger])
        score = m1.evaluate(X_unsw, y_unsw, y_unsw.shape[0], verbose=0)
        unsw_loss.append(score[0])
        score = m2.evaluate(X_nsl, y_nsl, y_nsl.shape[0], verbose=0)
        nsl_loss.append(score[0])

        m2.fit(X_nsl, y_nsl, batch_size, epochs=1, callbacks=[csv_logger])
        score = m1.evaluate(X_unsw, y_unsw, y_unsw.shape[0], verbose=0)
        unsw_loss.append(score[0])
        score = m2.evaluate(X_nsl, y_nsl, y_nsl.shape[0], verbose=0)
        nsl_loss.append(score[0])

    shared['unsw_loss'].append(unsw_loss)
    shared['nsl_loss'].append(nsl_loss)

    score = m1.evaluate(X_unsw, y_unsw, y_unsw.shape[0], verbose=0)
    logger.debug('UNSW train loss %.6f' % score[0])
    logger.info('UNSW train accu %.6f' % score[1])
    shared['unsw']['train'].append(score[1])
    score = m1.evaluate(X_unsw_test, y_unsw_test, y_unsw_test.shape[0],
                        verbose=0)
    logger.debug('UNSW test loss %.6f' % score[0])
    logger.info('UNSW test accu %.6f' % score[1])
    shared['unsw']['test'].append(score[1])

    score = m2.evaluate(X_nsl, y_nsl, y_nsl.shape[0], verbose=0)
    logger.debug('NSL train loss %.6f' % score[0])
    logger.info('NSL train accu %.6f' % score[1])
    shared['nsl']['train'].append(score[1])
    score = m2.evaluate(X_nsl_test, y_nsl_test, y_nsl_test.shape[0], verbose=0)
    logger.debug('NSL test loss %.6f' % score[0])
    logger.info('NSL test accu %.6f' % score[1])
    shared['nsl']['test'].append(score[1])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    root = 'SharedLayer/'
    logger = logging.getLogger('SharedLayer')
    hdlr = logging.FileHandler(root + 'accuracy.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    unsw_tens, unsw_inputs, X_unsw, X_unsw_test, y_unsw, y_unsw_test = \
        get_unsw_data()
    nsl_tens, nsl_inputs, X_nsl, X_nsl_test, y_nsl, y_nsl_test = \
        get_nsl_data()
    num_runs = 10
    hidden_configs = [[256, 480, 400]]
    for hidden in hidden_configs:
        shared = {'unsw': {'train': [], 'test': []},
                  'unsw_loss': [], 'nsl_loss': [],
                  'nsl': {'train': [], 'test': []}}
        logger.info('************************************************')
        logger.info('****  Start %d runs with shared config %s  ****'
                    % (num_runs, hidden_configs))
        logger.info('************************************************')
        for _ in range(num_runs):
            run_main(hidden)
        pprint(shared)
        output = open(root + 'result_runs%s_U%d' % (num_runs, hidden[1]), 'wb+')
        pickle.dump(shared, output)
        output.close()
