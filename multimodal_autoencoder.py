from keras.models import Model
from keras.layers import Dense, Input, concatenate, Flatten, Dropout
from keras import regularizers
from keras.layers import Embedding, BatchNormalization
from keras.callbacks import CSVLogger

from preprocess import unsw, nslkdd
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


def get_intermediate_output(model, layer_name, inputs, train_dict, test_dict):
    intermediate_layer_model = Model(inputs=inputs,
                                     outputs=model.get_layer(layer_name).output)
    EX = intermediate_layer_model.predict(train_dict)
    EX_test = intermediate_layer_model.predict(test_dict)

    return EX, EX_test


def modality_net_unsw(hidden, num_epochs, batch_size,
                      drop_prob=0.2, reg_beta=0.001):
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
    united = hidden[-1]
    merged_dim = 0
    merged_dim += build_embeddings(symbolic_features, integer_features,
                                   embeddings, large_discrete, merged_inputs,
                                   X, test_X, train_dict, test_dict, 'unsw')
    merged_dim += len(continuous_features)
    cont_component = build_continuous(continuous_features,
                                      merged_inputs, X, test_X,
                                      train_dict, test_dict, 'unsw')
    logger.debug('merge input_dim for UNSW-NB dataset = %s' % merged_dim)

    merge = concatenate(embeddings + large_discrete + [cont_component],
                        name='concate_features_unsw')
    h1 = Dense(hidden[0], activation='relu', name='hidden_unsw',
               kernel_regularizer=regularizers.l2(reg_beta))(merge)
    dropout = Dropout(drop_prob)(h1)
    bn = BatchNormalization(name='bn_unsw_1')(dropout)
    h2 = Dense(united, activation='sigmoid', name='unified_unsw',
               kernel_regularizer=regularizers.l2(reg_beta))(bn)
    sm = Dense(2, activation='softmax', name='output')(h2)

    model = Model(inputs=merged_inputs, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    csv_logger = CSVLogger('ModalityNets/mn_UNSW.history', append=True)
    history = model.fit(train_dict, {'output': y}, shuffle=True,
                        epochs=num_epochs, batch_size=batch_size,
                        callbacks=[csv_logger],
                        validation_data=(test_dict, test_y))
    logger.debug(history)
    score = model.evaluate(test_dict, test_y, test_y.shape[0], verbose=1)
    logger.debug('ModalityNet UNSW test loss\t%.6f' % score[0])
    logger.info('ModalityNet UNSW test accu\t%.6f' % score[1])

    EX, EX_test = get_intermediate_output(model, 'unified_unsw', merged_inputs,
                                          train_dict, test_dict)
    # model.save('ModalityNets/UNSW.h5')
    # np.savez('ModalityNets/unsw_EX.npy', train=EX, test=EX_test)
    return EX, EX_test, y, test_y


def process_nsl():
    dataset_names = ['NSLKDD/KDD%s.csv' % x for x in ['Train', 'Test']]
    feature_file = 'NSLKDD/feature_names.csv'
    headers, _, _, _ = nslkdd.get_feature_names(feature_file)
    symbolic_features = nslkdd.discovery_feature_volcabulary(dataset_names)
    integer_features = nslkdd.discovery_integer_map(feature_file, dataset_names)
    continuous_features = nslkdd.discovery_continuous_map(feature_file,
                                                          dataset_names)
    X, y = get_dataset(dataset_names[0], headers, 'nsl')
    X_test, y_test = get_dataset(dataset_names[1], headers, 'nsl')

    train_dict = dict()
    test_dict = dict()
    raw_inputs = []
    embeddings = []
    large_discrete = []
    merged_dim = 0
    merged_dim += build_embeddings(symbolic_features, integer_features,
                                   embeddings, large_discrete, raw_inputs,
                                   X, X_test, train_dict, test_dict, 'nsl')
    cont_component = build_continuous(continuous_features,
                                      raw_inputs, X, X_test,
                                      train_dict, test_dict, 'nsl')
    merged_dim += len(continuous_features)
    logger.debug('merge input_dim for NSLKDD dataset = %s' % merged_dim)
    merged_nsl = concatenate(embeddings + large_discrete + [cont_component],
                             name='concate_features_nsl')
    results = {'nsl': merged_nsl, 'nsl_dim': merged_dim,
               'raw_inputs': raw_inputs, 'X': X, 'y': y,
               'X_test': X_test, 'y_test': y_test}
    return results


def multimodal_autoencoder(U1, U2):
    unsw_dict = process_unsw()
    nsl_dict = process_nsl()

    unsw_inputs = unsw_dict['raw_inputs']
    unsw = unsw_dict['unsw']
    unsw_dim = unsw_dict['unsw_dim']
    X_unsw = unsw_dict['X']
    y_unsw = unsw_dict['y']
    X_test_unsw = unsw_dict['X_test']
    y_test_unsw = unsw_dict['y_test']

    nsl_inputs = nsl_dict['raw_inputs']
    nsl = nsl_dict['nsl']
    nsl_dim = nsl_dict['nsl_dim']
    X_nsl = nsl_dict['X']
    y_nsl = nsl_dict['y']
    X_test_nsl = nsl_dict['X_test']
    y_test_nsl = nsl_dict['y_test']

    h1_unsw = Dense(U1, activation='relu', name='h1_unsw')(unsw)
    h1_nsl = Dense(U1, activation='relu', name='h1_nsl')(nsl)
    bn_unsw = BatchNormalization(name='bn1_unsw')(h1_unsw)
    bn_nsl = BatchNormalization(name='bn1_nsl')(h1_nsl)
    drop_unsw = Dropout(0.2)(bn_unsw)
    drop_nsl = Dropout(0.2)(bn_nsl)

    shared_ae = Dense(U2, activation='sigmoid', name='shared')
    shared_unsw = shared_ae(drop_unsw)
    shared_nsl = shared_ae(drop_nsl)
    h3_unsw = Dense(unsw_dim, activation='sigmoid', name='h3_unsw')(shared_unsw)
    h3_nsl = Dense(nsl_dim, activation='sigmoid', name='h3_nsl')(shared_nsl)

    model_unsw = Model(inputs=unsw_inputs, outputs=h3_unsw)
    model_unsw.compile(optimizer='adam', loss='binary_crossentropy')

    model_nsl = Model(inputs=nsl_inputs, outputs=h3_nsl)
    model_nsl.compile(optimizer='adam', loss='binary_crossentropy')

    encoder_unsw = Model(inputs=unsw_inputs, outputs=shared_unsw)
    encoder_nsl = Model(inputs=nsl_inputs, outputs=shared_nsl)

    for _ in range(10):
        model_unsw.fit(X_unsw, X_unsw, epoch=1, batch_size=80)
        model_nsl.fit(X_nsl, X_nsl, epoch=1, batch_size=80)

    unified_X_unsw = encoder_unsw.predict(X_unsw)
    unified_X_test_unsw = encoder_unsw.predict(X_test_unsw)

    unified_X_nsl = encoder_nsl.predict(X_nsl)
    unified_X_test_nsl = encoder_nsl.predict(X_test_nsl)

    return {'X_unsw': unified_X_unsw, 'X_test_unsw': unified_X_test_unsw,
            'X_nsl': unified_X_nsl, 'X_test_nsl': unified_X_test_nsl,
            'y_unsw': y_unsw, 'y_test_unsw': y_test_unsw,
            'y_nsl': y_nsl, 'y_test_nsl': y_test_nsl}


def master_model(hidden, drop_prob=0.2, reg_beta=0.001):
    main_input = Input(shape=(hidden[0],), name='main_input')
    h1 = Dense(hidden[1], activation='sigmoid', name='h1',
               kernel_regularizer=regularizers.l2(reg_beta))(main_input)
    dropout = Dropout(drop_prob)(h1)
    bn = BatchNormalization(name='bn_1')(dropout)
    # h2 = Dense(320, activation='sigmoid', name='h2',
    # kernel_regularizer=regularizers.l2(reg_beta))(bn)
    # dropout = Dropout(0.2)(h2)
    # bn = BatchNormalization(name='bn_2')(dropout)
    sm = Dense(2, activation='softmax', name='output')(bn)
    model = Model(inputs=main_input, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def train_with_single(hidden, EX, y, EXTs, test_ys, idx, drop_prob,
                      num_epochs, batch_size,
                      names=['UNSW', 'NSL']):
    model = master_model(hidden, drop_prob=drop_prob, reg_beta=0.00)
    csv_logger = CSVLogger('ModalityNets/master_%s.history' % names[idx],
                           append=True)
    history = model.fit(EX, y, epochs=num_epochs, batch_size=batch_size,
                        shuffle=True, steps_per_epoch=None,
                        callbacks=[csv_logger],
                        validation_data=(EXTs[idx], test_ys[idx]))
    for (i, EXT) in enumerate(EXTs):
        score = model.evaluate(EXTs[i], test_ys[i], test_ys[i].shape[0])
        logger.debug('Master trained with %s dataset %s test loss\t%.6f' %
                     (names[idx], names[i], score[0]))
        logger.info('Master trained with %s dataset %s test accu\t%.6f' %
                    (names[idx], names[i], score[1]))
    return history


def train_with_both(hidden, EXs, ys, EXTs, test_ys, drop_prob,
                    num_epochs, batch_size,
                    names=['UNSW', 'NSL']):
    EX = np.concatenate(EXs, axis=0)
    Ey = np.concatenate(ys, axis=0)
    model = master_model(hidden, drop_prob=drop_prob, reg_beta=0.00)
    csv_logger = CSVLogger('ModalityNets/master_BOTH.history', append=True)
    history = model.fit(EX, Ey, epochs=num_epochs,
                        batch_size=batch_size, shuffle=True,
                        steps_per_epoch=None, callbacks=[csv_logger])
    for (i, EXT) in enumerate(EXTs):
        score = model.evaluate(EXTs[i], test_ys[i], test_ys[i].shape[0])
        logger.debug('Master trained with BOTH dataset %s test loss\t%.6f' %
                     (names[i], score[0]))
        logger.info('Master trained with BOTH dataset %s test accu\t%.6f' %
                    (names[i], score[1]))

    return history


def run_master(united, h):
    num_epochs = 40
    batch_size = 80
    beta = 0.00
    drop_prob = 0.2
    hidden_unsw = [256, united]
    hidden_nsl = [256, united]
    hidden_master = [united, h]
    logger.info('Network Config: %s %s %s' % (hidden_unsw,
                                              hidden_nsl, hidden_master))
    EX1, EXT1, y1, test_y1 = modality_net_unsw(hidden_unsw, num_epochs,
                                               batch_size, drop_prob, beta)
    EX2, EXT2, y2, test_y2 = modality_net_nsl(hidden_nsl, num_epochs,
                                              batch_size, drop_prob, beta)
    train_with_single(hidden_master, EX1, y1,
                      [EXT1, EXT2], [test_y1, test_y2], 0, drop_prob,
                      num_epochs, batch_size)
    train_with_single(hidden_master, EX2, y2,
                      [EXT1, EXT2], [test_y1, test_y2], 1, drop_prob,
                      num_epochs, batch_size)
    train_with_both(hidden_master, [EX1, EX2], [y1, y2],
                    [EXT1, EXT2], [test_y1, test_y2], drop_prob,
                    num_epochs, batch_size)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('modality_nets')
    hdlr = logging.FileHandler('ModalityNets/accuracy.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    # layer_sizes = [180, 240, 270, 360, 480, 540]
    layer_sizes = [480]
    num_runs = 10
    for _ in range(num_runs):
        for united in layer_sizes:
            for h in layer_sizes:
                run_master(united, h)
