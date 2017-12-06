from keras.models import Model
from keras.layers import Dense, Input, merge
# from keras.layers import Dropout
from keras import regularizers
from keras.layers import BatchNormalization
from keras.callbacks import CSVLogger
# from keras import initializers
from lightgbm import LGBMClassifier
import pickle
import tensorflow as tf
from keras.backend import tensorflow_backend as K

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


def multimodal_autoencoder(unsw_dim, nsl_dim, H1, U, sparse=0.00):
    unsw = Input(shape=(unsw_dim, ), name='input_unsw')
    nsl = Input(shape=(nsl_dim, ), name='input_nsl')

    h1_unsw = Dense(H1, activation='relu', name='h1_unsw',
                    activity_regularizer=regularizers.l1(sparse))(unsw)
    h1_nsl = Dense(H1, activation='relu', name='h1_nsl',
                   activity_regularizer=regularizers.l1(sparse))(nsl)
    h1_unsw = BatchNormalization(name='bn1_unsw')(h1_unsw)
    h1_nsl = BatchNormalization(name='bn1_nsl')(h1_nsl)

    shared_ae = Dense(U, activation='relu', name='shared',
                      activity_regularizer=regularizers.l1(sparse))
    shared_unsw = shared_ae(h1_unsw)
    shared_nsl = shared_ae(h1_nsl)
    bns_unsw = BatchNormalization(name='bn2_unsw')(shared_unsw)
    bns_nsl = BatchNormalization(name='bn2_nsl')(shared_nsl)

    h3_unsw = Dense(H1, activation='relu', name='h3_unsw')(bns_unsw)
    h3_nsl = Dense(H1, activation='relu', name='h3_nsl')(bns_nsl)
    h3_unsw = BatchNormalization(name='bn3_unsw')(h3_unsw)
    h3_nsl = BatchNormalization(name='bn3_nsl')(h3_nsl)

    h4_unsw = Dense(unsw_dim, activation='relu', name='h4_unsw')(h3_unsw)
    h4_nsl = Dense(nsl_dim, activation='relu', name='h4_nsl')(h3_nsl)

    model_unsw = Model(inputs=unsw, outputs=h4_unsw)
    model_unsw.compile(optimizer='adadelta', loss='binary_crossentropy')
    model_unsw.summary()
    model_nsl = Model(inputs=nsl, outputs=h4_nsl)
    model_nsl.compile(optimizer='adadelta', loss='binary_crossentropy')
    model_nsl.summary()
    encoder_unsw = Model(inputs=unsw, outputs=bns_unsw)
    encoder_nsl = Model(inputs=nsl, outputs=bns_nsl)

    return model_unsw, model_nsl, encoder_unsw, encoder_nsl


def build_attention_model(input_dim, nb_classes):
    inputs = Input(shape=(input_dim[0],))

    # ATTENTION PART STARTS HERE
    attention_probs = Dense(input_dim[0], activation='softmax', name='attention_vec')(inputs)
    attention_mul = merge([inputs, attention_probs], output_shape=nb_classes, name='attention_mul', mode='mul')
    # ATTENTION PART FINISHES HERE

    attention_mul = Dense(64)(attention_mul)
    output = Dense(units=nb_classes, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)

    return model


def supervised_shared(unsw_dict, nsl_dict, H1, U, num_epochs, batch_size, beta):
    load = False
    if load:
        with open(r"SharedAutoEncoder/datasets.p", "rb") as i:
            EX_unsw, EX_unsw_test, EX_nsl, EX_nsl_test = pickle.load(i)
    else:
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
        for x in range(num_epochs):
            print("Epoch:",x)
            model_unsw.fit(X_unsw, X_unsw, epochs=2, batch_size=batch_size)
            model_nsl.fit(X_nsl, X_nsl, epochs=2, batch_size=batch_size)

        # Get the shared representation of both datasets
        EX_unsw = encoder_unsw.predict(X_unsw)
        EX_unsw_test = encoder_unsw.predict(X_unsw_test)

        EX_nsl = encoder_nsl.predict(X_nsl)
        EX_nsl_test = encoder_nsl.predict(X_nsl_test)

        with open(r"SharedAutoEncoder/datasets.p", "wb") as o:
            pickle.dump((EX_unsw, EX_unsw_test, EX_nsl, EX_nsl_test), o)

        # Get accu5(unsw) and accu5(nsl)
        #EX_concat = np.concatenate((EX_unsw, EX_nsl), axis=0)
        #y_concat = np.concatenate((y_unsw, y_nsl), axis=0)

    #model = build_attention_model(EX_unsw.shape[1], 2)

    model = LGBMClassifier(n_jobs=8
                           , max_depth=11
                           , num_leaves=302
                           , learning_rate=0.1
                           , n_estimators=500
                           # ,max_bin=15
                           #, colsample_bytree=0.8
                           #, subsample=0.8
                           #, min_child_weight=6
                           )


    #model.fit(EX_concat, y_concat[:,0])
    logger.info("Training lgbm model on NSL unified representation")
    model.fit(EX_nsl, y_nsl[:, 0], early_stopping_rounds=3, eval_set=(EX_nsl_test, y_nsl_test[:, 0]), verbose=False)
    logger.info("Shared model NSL train acc:\t%.6f" % model.score(EX_nsl, y_nsl[:, 0]))
    logger.info("Shared model NSL test acc:\t%.6f" % model.score(EX_nsl_test, y_nsl_test[:, 0]))

    logger.info("Training lgbm model on UNSW unified representation")
    model.fit(EX_unsw, y_unsw[:, 0], early_stopping_rounds=3, eval_set=(EX_unsw_test, y_unsw_test[:, 0]), verbose=False)
    #model.fit(EX_nsl, y_nsl[:, 0], early_stopping_rounds=3, eval_set=(EX_nsl_test, y_nsl_test[:, 0]), verbose=False)


    logger.info("Shared model UNSW train acc:\t%.6f" % model.score(EX_unsw, y_unsw[:,0]))
    logger.info("Shared model UNSW test acc:\t%.6f" % model.score(EX_unsw_test, y_unsw_test[:,0]))
    logger.info("Shared model NSL train acc:\t%.6f" % model.score(EX_nsl, y_nsl[:,0]))
    logger.info("Shared model NSL test acc:\t%.6f" % model.score(EX_nsl_test, y_nsl_test[:,0]))
    

def run_master(unsw_dict, nsl_dict, H1, U):
    num_epochs = 1
    batch_size = 64
    beta = 0.01
    supervised_shared(unsw_dict, nsl_dict, H1, U, num_epochs, batch_size, beta)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('SharedAE')
    hdlr = logging.FileHandler('SharedAutoEncoder/accuracy.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    unsw_dict = process_unsw()
    nsl_dict = process_nsl()

    # layer_sizes = [128, 240, 320, 400]
    layer_sizes = [512]
    num_runs = 1
    for H1 in layer_sizes:
        logger.info('********************************************************')
        logger.info('**** Start %d runs with various layer config %d *******'
                    % (num_runs, H1))
        logger.info('********************************************************')
        for i in range(num_runs):
            logger.info('*** Run index %d ***' % i)
            run_master(unsw_dict, nsl_dict, H1, H1 * 2)
            tf.reset_default_graph()
