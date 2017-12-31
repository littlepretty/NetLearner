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


def get_intermediate_output(model, layer_name, inputs, train_dict, test_dict):
    intermediate_layer_model = Model(inputs=inputs,
                                     outputs=model.get_layer(layer_name).output)
    EX = intermediate_layer_model.predict(train_dict)
    EX_test = intermediate_layer_model.predict(test_dict)

    return EX, EX_test


def modality_net_unsw(hidden):
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
    logger.debug('merge input_dim for UNSW-NB dataset = %s' % merged_dim)

    merge = concatenate(embeddings + large_discrete + [cont_component],
                        name='concate_features_unsw')
    h1 = Dense(hidden[0], activation='relu', name='hidden_unsw')(merge)
    dropout = Dropout(drop_prob)(h1)
    h2 = Dense(hidden[1], activation='relu', name='unified_unsw')(dropout)

    bn = BatchNormalization(name='bn_unified_unsw')(h2)
    h3 = Dense(hidden[2], activation='sigmoid', name='sigmoid')(bn)
    sm = Dense(2, activation='softmax', name='output')(h3)

    model = Model(inputs=merged_inputs, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    csv_logger = CSVLogger(root + 'modnet_unsw.history', append=True)
    history = model.fit(train_dict, {'output': y}, batch_size, num_epochs,
                        callbacks=[csv_logger])
    modnet['unsw_loss'].append(history.history['loss'])
    score = model.evaluate(train_dict, y, y.shape[0])
    logger.debug('modnet[unsw] train loss\t%.6f' % score[0])
    logger.info('modenet[unsw] train accu\t%.6f' % score[1])
    modnet['unsw']['train'].append(score[1])

    score = model.evaluate(test_dict, test_y, test_y.shape[0])
    logger.debug('modnet[unsw] test loss\t%.6f' % score[0])
    logger.info('modenet[unsw] test accu\t%.6f' % score[1])
    modnet['unsw']['test'].append(score[1])

    EX, EX_test = get_intermediate_output(model, 'unified_unsw', merged_inputs,
                                          train_dict, test_dict)
    # model.save('ModalityNets/UNSW.h5')
    # np.savez('ModalityNets/unsw_EX.npy', train=EX, test=EX_test)
    return EX, EX_test, y, test_y


def modality_net_nsl(hidden):
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
    logger.debug('merge input_dim for NSLKDD dataset = %s' % merged_dim)

    merge = concatenate(embeddings + large_discrete + [cont_component],
                        name='concate_features_nsl')
    h1 = Dense(hidden[0], activation='relu', name='hidden_nsl')(merge)
    dropout = Dropout(drop_prob)(h1)
    h2 = Dense(hidden[1], activation='relu', name='unified_nsl')(dropout)

    bn = BatchNormalization(name='bn_unified_nsl')(h2)
    h3 = Dense(hidden[2], activation='sigmoid', name='sigmoid')(bn)
    sm = Dense(2, activation='softmax', name='output')(h3)

    model = Model(inputs=merged_inputs, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    csv_logger = CSVLogger(root + 'modnet_nsl.history', append=True)
    history = model.fit(train_dict, {'output': y}, batch_size,
                        num_epochs, callbacks=[csv_logger])
    modnet['nsl_loss'].append(history.history['loss'])
    score = model.evaluate(train_dict, y, y.shape[0])
    logger.debug('modnet[nsl] train loss\t%.6f' % score[0])
    logger.info('modenet[nsl] train accu\t%.6f' % score[1])
    modnet['nsl']['train'].append(score[1])

    score = model.evaluate(test_dict, test_y, test_y.shape[0])
    logger.debug('modnet[nsl] test loss\t%.6f' % score[0])
    logger.info('modenet[nsl] test accu\t%.6f' % score[1])
    modnet['nsl']['test'].append(score[1])

    EX, EX_test = get_intermediate_output(model, 'unified_nsl', merged_inputs,
                                          train_dict, test_dict)
    # model.save('ModalityNets/NSL.h5')
    # np.savez('ModalityNets/nsl_EX.npy', train=EX, test=EX_test)
    return EX, EX_test, y, test_y


def unified_model(hidden, drop_prob=0.2, reg_beta=0.001):
    main_input = Input(shape=(hidden[0],), name='main_input')
    bn = BatchNormalization(name='bn_unified')(main_input)
    h1 = Dense(hidden[1], activation='sigmoid', name='h1')(bn)
    sm = Dense(2, activation='softmax', name='output')(h1)
    model = Model(inputs=main_input, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def both_dataset(hidden, EXs, ys, EXTs, test_ys, names=['unsw', 'nsl']):
    model = unified_model(hidden, drop_prob=drop_prob, reg_beta=0.00)
    unsw_loss, nsl_loss = [], []
    for _ in range(num_epochs):
        history = model.fit(EXs[0], ys[0], batch_size, 1)
        unsw_loss.append(history.history['loss'])
        history = model.fit(EXs[1], ys[1], batch_size, 1)
        nsl_loss.append(history.history['loss'])

    unified['unsw_loss'].append(unsw_loss)
    unified['nsl_loss'].append(nsl_loss)

    for (i, EXT) in enumerate(EXTs):
        score = model.evaluate(EXs[i], ys[i], ys[i].shape[0])
        logger.debug('unified[%s] train loss\t%.6f' % (names[i], score[0]))
        logger.info('unified[%s] train accu\t%.6f' % (names[i], score[1]))
        unified[names[i]]['train'].append(score[1])

        score = model.evaluate(EXTs[i], test_ys[i], test_ys[i].shape[0])
        logger.debug('unified[%s] test loss\t%.6f' % (names[i], score[0]))
        logger.info('unified[%s] test accu\t%.6f' % (names[i], score[1]))
        unified[names[i]]['test'].append(score[1])


def run_master(unsw_config, nsl_config):
    EX1, EXT1, y1, test_y1 = modality_net_unsw(unsw_config)
    EX2, EXT2, y2, test_y2 = modality_net_nsl(nsl_config)

    hidden_master = [unsw_config[-2], unsw_config[-1]]
    logger.info('Unified Net Config: %s' % hidden_master)
    both_dataset(hidden_master, [EX1, EX2], [y1, y2],
                 [EXT1, EXT2], [test_y1, test_y2])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    root = 'ModalityNets/'
    logger = logging.getLogger('ModalityNets')
    hdlr = logging.FileHandler(root + 'accuracy.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    h_front = [[640, 480]]
    h_unified = [512]
    h_cls = [400]
    num_runs = 30
    num_epochs = 36
    batch_size = 160
    beta = 0.00
    drop_prob = 0.2
    for (i, u) in enumerate(h_unified):
        unsw_config = [h_front[i][0], u, h_cls[i]]
        nsl_config = [h_front[i][1], u, h_cls[i]]
        modnet = {'unsw': {'test': [], 'train': []},
                  'unsw_loss': [], 'nsl_loss': [],
                  'nsl': {'test': [], 'train': []}}
        unified = {'unsw': {'test': [], 'train': []},
                   'unsw_loss': [], 'nsl_loss': [],
                   'nsl': {'test': [], 'train': []}}
        logger.info('**********************************************')
        logger.info('****  Start %d runs with config %s %s  ****'
                    % (num_runs, unsw_config, nsl_config))
        logger.info('**********************************************')
        for _ in range(num_runs):
            run_master(unsw_config, nsl_config)

        result = {'modnet': modnet, 'unified': unified,
                  'epochs': num_epochs, 'batch_size': batch_size,
                  'unsw_config': unsw_config, 'nsl_config': nsl_config,
                  'dropout': drop_prob, 'beta': beta}
        pprint(result)
        output = open(root + 'result_runs%d_U%d.pkl' % (num_runs, u), 'wb+')
        pickle.dump(result, output)
        output.close()
