from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras import regularizers
from keras.layers import BatchNormalization
from keras.callbacks import CSVLogger

from preprocess import unsw, nslkdd
from netlearner.utils import permutate_dataset, min_max_scale

import numpy as np
import logging


def process_unsw():
    unsw.generate_dataset(True)
    raw_X_train = np.load('UNSW/train_dataset.npy')
    y_train = np.load('UNSW/train_labels.npy')
    raw_X_test = np.load('UNSW/test_dataset.npy')
    y_test = np.load('UNSW/test_labels.npy')
    [X_train, _, X_test] = min_max_scale(raw_X_train, None, raw_X_test)
    permutate_dataset(X_train, y_train)
    permutate_dataset(X_test, y_test)

    print('Training set', X_train.shape, y_train.shape)
    print('Test set', X_test.shape, y_test.shape)
    return {'X': X_train, 'y': y_train, 'X_test': X_test, 'y_test': y_test}


def process_nsl():
    nslkdd.generate_datasets(binary_label=True)
    raw_X_train = np.load('NSLKDD/train_dataset.npy')
    y_train = np.load('NSLKDD/train_labels.npy')
    raw_X_test = np.load('NSLKDD/test_dataset.npy')
    y_test = np.load('NSLKDD/test_labels.npy')
    [X_train, _, X_test] = min_max_scale(raw_X_train, None, raw_X_test)
    permutate_dataset(X_train, y_train)
    permutate_dataset(X_test, y_test)

    print('Training set', X_train.shape, y_train.shape)
    print('Test set', X_test.shape, y_test.shape)
    return {'X': X_train, 'y': y_train, 'X_test': X_test, 'y_test': y_test}


def single_encoder(feature_dim, H1, U):
    input_layer = Input(shape=(feature_dim, ), name='unsw')

    h1 = Dense(H1, activation='relu', name='h1')(input_layer)
    bn1 = BatchNormalization(name='bn1')(h1)

    encoding = Dense(U, activation='sigmoid', name='encoding')(bn1)
    bn2 = BatchNormalization(name='bn2')(encoding)

    h3 = Dense(H1, activation='sigmoid', name='h3')(bn2)
    bn3 = BatchNormalization(name='bn3')(h3)

    h4 = Dense(feature_dim, activation='sigmoid', name='h4')(bn3)

    model = Model(inputs=input_layer, outputs=h4)
    model.compile(optimizer='adam', loss='binary_crossentropy')

    encoder = Model(inputs=input_layer, outputs=encoding)  # or bn2
    return model, encoder


def multimodal_autoencoder(unsw_dim, nsl_dim, H1, U):
    unsw = Input(shape=(unsw_dim, ), name='input_unsw')
    nsl = Input(shape=(nsl_dim, ), name='input_nsl')

    h1_unsw = Dense(H1, activation='relu', name='h1_unsw')(unsw)
    h1_nsl = Dense(H1, activation='relu', name='h1_nsl')(nsl)
    bn1_unsw = BatchNormalization(name='bn1_unsw')(h1_unsw)
    bn1_nsl = BatchNormalization(name='bn1_nsl')(h1_nsl)

    shared_ae = Dense(U, activation='sigmoid', name='shared')
    shared_unsw = shared_ae(bn1_unsw)
    shared_nsl = shared_ae(bn1_nsl)
    bn2_unsw = BatchNormalization(name='bn2_unsw')(shared_unsw)
    bn2_nsl = BatchNormalization(name='bn2_nsl')(shared_nsl)

    h3_unsw = Dense(H1, activation='sigmoid', name='h3_unsw')(bn2_unsw)
    h3_nsl = Dense(H1, activation='sigmoid', name='h3_nsl')(bn2_nsl)
    bn3_unsw = BatchNormalization(name='bn3_unsw')(h3_unsw)
    bn3_nsl = BatchNormalization(name='bn3_nsl')(h3_nsl)

    h4_unsw = Dense(unsw_dim, activation='sigmoid', name='h4_unsw')(bn3_unsw)
    h4_nsl = Dense(nsl_dim, activation='sigmoid', name='h4_nsl')(bn3_nsl)

    model_unsw = Model(inputs=unsw, outputs=h4_unsw)
    model_unsw.compile(optimizer='adam', loss='binary_crossentropy')
    model_unsw.summary()
    model_nsl = Model(inputs=nsl, outputs=h4_nsl)
    model_nsl.compile(optimizer='adam', loss='binary_crossentropy')
    model_nsl.summary()
    encoder_unsw = Model(inputs=unsw, outputs=shared_unsw)
    encoder_nsl = Model(inputs=nsl, outputs=shared_nsl)

    return model_unsw, model_nsl, encoder_unsw, encoder_nsl


def linear_model(feature_dim, reg_beta=0.001):
    main_input = Input(shape=(feature_dim, ), name='main_input')
    sm = Dense(2, activation='softmax', name='h1',
               kernel_regularizer=regularizers.l2(reg_beta))(main_input)
    model = Model(inputs=main_input, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def train_linear_model(X, y, X_test, y_test, num_epochs, batch_size, beta):
    feature_dim = X.shape[1]
    classifier = linear_model(feature_dim, beta)
    classifier.fit(X, y, batch_size=batch_size, epochs=num_epochs)
    scores = classifier.evaluate(X_test, y_test, batch_size=X_test.shape[0])
    return scores


def train_single_encoder(X, X_test, H1, U, num_epochs, batch_size, name):
    feature_dim = X.shape[1]
    model, encoder = single_encoder(feature_dim, H1, U)
    csv_logger = CSVLogger('SharedAutoEncoder/ae_%s.history' % name,
                           append=True)
    model.fit(X, X, epochs=num_epochs, batch_size=batch_size,
              callbacks=[csv_logger])
    EX = encoder.predict(X)
    EX_test = encoder.predict(X_test)

    return EX, EX_test


def supervised_single(unsw_dict, nsl_dict, H1, U, num_epochs, batch_size, beta):
    logger.info('Using Single AE with Linear Classifier')
    X_unsw = unsw_dict['X']
    X_unsw_test = unsw_dict['X_test']
    y_unsw = unsw_dict['y']
    y_unsw_test = unsw_dict['y_test']
    X_nsl = nsl_dict['X']
    X_nsl_test = nsl_dict['X_test']
    y_nsl = nsl_dict['y']
    y_nsl_test = nsl_dict['y_test']
    EX_unsw, EX_unsw_test = train_single_encoder(X_unsw, X_unsw_test,
                                                 H1, U, num_epochs,
                                                 batch_size, 'unsw')
    EX_nsl, EX_nsl_test = train_single_encoder(X_nsl, X_nsl_test,
                                               H1, U, num_epochs,
                                               batch_size, 'nsl')
    # Get accu6(unsw) and accu6(nsl)
    EX_concat = np.concatenate((EX_unsw, EX_nsl), axis=0)
    y_concat = np.concatenate((y_unsw, y_nsl), axis=0)
    print(EX_concat.shape, y_concat.shape)
    scores6U = train_linear_model(EX_concat, y_concat,
                                  EX_unsw_test, y_unsw_test,
                                  num_epochs, batch_size, beta)
    logger.info('Trained on concat unshared-encoding, UNSW accu6\t%.6f'
                % scores6U[1])
    scores6N = train_linear_model(EX_concat, y_concat,
                                  EX_nsl_test, y_nsl_test,
                                  num_epochs, batch_size, beta)
    logger.info('Trained on concat unshared-encoding, NSL accu6\t%.6f'
                % scores6N[1])
    # Get accu3
    EX_unsw, EX_unsw_test = train_single_encoder(X_unsw, X_unsw_test,
                                                 H1, U, num_epochs,
                                                 batch_size, 'unsw')
    scores3 = train_linear_model(EX_unsw, y_unsw,
                                 EX_unsw_test, y_unsw_test,
                                 num_epochs, batch_size, beta)
    logger.info('Trained on single UNSW-encoding, accu3\t%.6f' % scores3[1])
    # Get accu1
    EX_nsl, EX_nsl_test = train_single_encoder(X_nsl, X_nsl_test,
                                               H1, U, num_epochs,
                                               batch_size, 'nsl')
    scores1 = train_linear_model(EX_nsl, y_nsl,
                                 EX_nsl_test, y_nsl_test,
                                 num_epochs, batch_size, beta)
    logger.info('Trained on single NSL-encoding, accu1\t%.6f' % scores1[1])
    return {'accu1': scores1[1], 'accu3': scores3[1],
            'accu6(UNSL)': scores6U[1], 'accu6(NSL)': scores6N[1]}


def supervised_shared(unsw_dict, nsl_dict, H1, U, num_epochs, batch_size, beta):
    logger.info('Using Shared AE with Linear Classifier')
    X_unsw = unsw_dict['X']
    X_unsw_test = unsw_dict['X_test']
    y_unsw = unsw_dict['y']
    y_unsw_test = unsw_dict['y_test']

    X_nsl = nsl_dict['X']
    X_nsl_test = nsl_dict['X_test']
    y_nsl = nsl_dict['y']
    y_nsl_test = nsl_dict['y_test']
    unsw_dim = X_unsw.shape[1]
    nsl_dim = X_nsl.shape[1]

    model_unsw, model_nsl, encoder_unsw, encoder_nsl = multimodal_autoencoder(
        unsw_dim, nsl_dim, H1, U)
    for _ in range(num_epochs * 3):
        model_unsw.fit(X_unsw, X_unsw, epochs=1, batch_size=80)
        model_nsl.fit(X_nsl, X_nsl, epochs=1, batch_size=80)

    # Get the shared representation of both datasets
    EX_unsw = encoder_unsw.predict(X_unsw)
    EX_unsw_test = encoder_unsw.predict(X_unsw_test)

    EX_nsl = encoder_nsl.predict(X_nsl)
    EX_nsl_test = encoder_nsl.predict(X_nsl_test)

    # Get accu5(unsw) and accu5(nsl)
    EX_concat = np.concatenate((EX_unsw, EX_nsl), axis=0)
    y_concat = np.concatenate((y_unsw, y_nsl), axis=0)
    scores5U = train_linear_model(EX_concat, y_concat,
                                  EX_unsw_test, y_unsw_test,
                                  num_epochs, batch_size, beta)
    logger.info('Trained on concat shared-encoding, UNSW accu5\t%.6f'
                % scores5U[1])
    scores5N = train_linear_model(EX_concat, y_concat,
                                  EX_nsl_test, y_nsl_test,
                                  num_epochs, batch_size, beta)
    logger.info('Trained on concat shared-encoding, NSL accu5\t%.6f'
                % scores5N[1])
    # Get accu4
    scores4 = train_linear_model(EX_unsw, y_unsw,
                                 EX_unsw_test, y_unsw_test,
                                 num_epochs, batch_size, beta)
    logger.info('Trained on shared UNSW-encoding, accu4\t%.6f' % scores4[1])
    # Get accu2
    scores2 = train_linear_model(EX_nsl, y_nsl,
                                 EX_nsl_test, y_nsl_test,
                                 num_epochs, batch_size, beta)
    logger.info('Trained on shared NSL-encoding, accu2\t%.6f' % scores2[1])
    return {'accu2': scores2[1], 'accu4': scores4[1],
            'accu5(UNSL)': scores5U[1], 'accu5(NSL)': scores5N[1]}


def run_master(unsw_dict, nsl_dict, H1, U):
    num_epochs = 12
    batch_size = 100
    beta = 0.00
    logger.info('Network Config: %s %s %s %s' % (H1, U, U, H1))
    supervised_single(unsw_dict, nsl_dict, H1, U, num_epochs, batch_size, beta)
    supervised_shared(unsw_dict, nsl_dict, H1, U, num_epochs, batch_size, beta)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('shared_autoencoder')
    hdlr = logging.FileHandler('SharedAutoEncoder/accuracy.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    unsw_dict = process_unsw()
    nsl_dict = process_nsl()

    # layer_sizes = [180, 240, 270, 360, 480, 540]
    layer_sizes = [480]
    num_runs = 1
    for _ in range(num_runs):
        for U in layer_sizes:
            for H1 in layer_sizes:
                run_master(unsw_dict, nsl_dict, H1, U)
