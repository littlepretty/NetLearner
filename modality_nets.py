from keras.models import Model
from keras.layers import Dense, Input, concatenate, Flatten, Dropout
from keras import regularizers
from keras.layers import Embedding, BatchNormalization

from preprocess import unsw, nslkdd
from netlearner.utils import permutate_dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

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
        logger.info('Dimension of %s E=%s and V=%s' % (name, dim_E, dim_V))
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
            logger.info('Dimension of %s E=%s and V=%s' % (name, dim_E, dim_V))
            temp = Embedding(output_dim=dim_E, input_dim=dim_V,
                             input_length=1,
                             name='embed_%s' % feature_name)(column)
            temp = Flatten(name='flat_%s' % feature_name)(temp)
            embeddings.append(temp)
            merged_dim += dim_E
        else:
            large_discrete.append(column)
            logger.info('Large feature %s is treated as continuous'
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


def modality_net_unsw(num_epochs, batch_size, reg_beta):
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
    logger.info('merge input_dim for this dataset = %s' % merged_dim)

    merge = concatenate(embeddings + large_discrete + [cont_component],
                        name='concate_features_unsw')
    h1 = Dense(400, activation='relu', name='hidden_unsw',
               kernel_regularizer=regularizers.l2(reg_beta))(merge)
    dropout = Dropout(0.2)(h1)
    bn = BatchNormalization(name='bn_unsw_1')(dropout)
    h2 = Dense(500, activation='relu', name='unified_unsw',
               kernel_regularizer=regularizers.l2(reg_beta))(bn)
    dropout = Dropout(0.2)(h2)
    bn = BatchNormalization(name='bn_nsl_2')(dropout)
    h3 = Dense(600, activation='relu', name='separate_nsl',
               kernel_regularizer=regularizers.l2(reg_beta))(bn)
    sm = Dense(2, activation='softmax', name='output')(h3)

    model = Model(inputs=merged_inputs, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_dict, {'output': y}, shuffle=True,
                        epochs=num_epochs, batch_size=batch_size,
                        validation_data=(test_dict, test_y))
    logger.debug(history)
    score = model.evaluate(test_dict, test_y, test_y.shape[0], verbose=1)
    logger.info('UNSW alone test loss %.6f' % score[0])
    logger.info('UNSW alone test accu %.6f' % score[1])

    model.save('ModalityNets/UNSW.h5')
    layer_name = 'unified_unsw'
    intermediate_layer_model = Model(inputs=merged_inputs,
                                     outputs=model.get_layer(layer_name).output)
    EX = intermediate_layer_model.predict(train_dict)
    EX_test = intermediate_layer_model.predict(test_dict)
    np.savez('ModalityNets/unsw_EX.npy', train=EX, test=EX_test)

    return EX, EX_test, y, test_y


def modality_net_nsl(num_epochs, batch_size, reg_beta=0.001):
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
    logger.info('merge input_dim for this dataset = %s' % merged_dim)

    merge = concatenate(embeddings + large_discrete + [cont_component],
                        name='concate_features_nsl')
    h1 = Dense(400, activation='relu', name='hidden_nsl',
               kernel_regularizer=regularizers.l2(reg_beta))(merge)
    dropout = Dropout(0.2)(h1)
    bn = BatchNormalization(name='bn_nsl_1')(dropout)
    h2 = Dense(500, activation='relu', name='unified_nsl',
               kernel_regularizer=regularizers.l2(reg_beta))(bn)
    dropout = Dropout(0.2)(h2)
    bn = BatchNormalization(name='bn_nsl_2')(dropout)
    h3 = Dense(600, activation='relu', name='separate_nsl',
               kernel_regularizer=regularizers.l2(reg_beta))(bn)
    sm = Dense(2, activation='softmax', name='output')(h3)

    model = Model(inputs=merged_inputs, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_dict, {'output': y}, shuffle=True,
                        epochs=num_epochs, batch_size=batch_size,
                        validation_data=(test_dict, test_y))
    logger.debug(history)
    score = model.evaluate(test_dict, test_y, test_y.shape[0], verbose=1)
    logger.info('NSL alone test loss %.6f' % score[0])
    logger.info('NSL alone test accu %.6f' % score[1])

    model.save('ModalityNets/NSL.h5')

    layer_name = 'unified_nsl'
    intermediate_layer_model = Model(inputs=merged_inputs,
                                     outputs=model.get_layer(layer_name).output)
    EX = intermediate_layer_model.predict(train_dict)
    EX_test = intermediate_layer_model.predict(test_dict)
    np.savez('ModalityNets/nsl_EX.npy', train=EX, test=EX_test)

    return EX, EX_test, y, test_y


def master_model(reg_beta=0.001):
    main_input = Input(shape=(500,), name='main_input')
    h1 = Dense(600, activation='relu', name='h1',
               kernel_regularizer=regularizers.l2(reg_beta))(main_input)
    dropout = Dropout(0.2)(h1)
    bn = BatchNormalization(name='bn')(dropout)
    h2 = Dense(800, activation='relu', name='h2',
               kernel_regularizer=regularizers.l2(reg_beta))(bn)
    sm = Dense(2, activation='softmax', name='output')(h2)
    model = Model(inputs=main_input, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
EX1, EXT1, y1, test_y1 = modality_net_unsw(10, 80, 0.001)
EX2, EXT2, y2, test_y2 = modality_net_nsl(10, 80, 0.001)
EX = np.concatenate((EX1, EX2), axis=0)
EXT = np.concatenate((EXT1, EXT2), axis=0)
Ey = np.concatenate((y1, y2), axis=0)
EyT = np.concatenate((test_y1, test_y2), axis=0)
dataset = dict()
dataset['train'], dataset['train_label'] = permutate_dataset(EX, Ey)
dataset['test'], dataset['test_label'] = permutate_dataset(EXT, EyT)
model = master_model(0.001)
num_epochs = 10
batch_size = 80
history = model.fit(dataset['train'], dataset['train_label'],
                    epochs=num_epochs, batch_size=batch_size, shuffle=True,
                    steps_per_epoch=None)
score = model.evaluate(EXT1, test_y1, test_y1.shape[0], verbose=1)
logger.info('UNSW test loss %.6f' % score[0])
logger.info('UNSW test accu %.6f' % score[1])

score = model.evaluate(EXT2, test_y2, test_y2.shape[0], verbose=1)
logger.info('NSL test loss %.6f' % score[0])
logger.info('NSL test accu %.6f' % score[1])
