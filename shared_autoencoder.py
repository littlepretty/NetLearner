from keras.models import Model
from keras.layers import Dense, Input
# from keras.layers import Dropout
from keras import regularizers
from keras.layers import BatchNormalization
# from keras.callbacks import CSVLogger
# from keras import initializers
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.backend import tensorflow_backend as K
from embedding_merger import get_unsw_data, get_nsl_data
from pprint import pprint
import logging
import numpy as np
import pickle


def multicore_session():
    config = tf.ConfigProto(intra_op_parallelism_threads=32,
                            inter_op_parallelism_threads=32,
                            allow_soft_placement=True,
                            log_device_placement=False,
                            device_count={'CPU': 64})
    session = tf.Session(config=config)
    K.set_session(session)


def classifier_model(feature_dim, hidden):
    main_input = Input(shape=(feature_dim, ), name='main_input')
    bn = BatchNormalization(name='bn')(main_input)
    h1 = Dense(hidden, activation='sigmoid', name='h1')(bn)
    sm = Dense(2, activation='softmax', name='output',
               kernel_regularizer=regularizers.l2(beta))(h1)
    model = Model(inputs=main_input, outputs=sm)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model


def single_encoder_model(input_dim, hidden):
    input_layer = Input(shape=(input_dim, ), name='input')
    h1 = Dense(hidden[0], activation='relu', name='h1')(input_layer)
    # bn1 = BatchNormalization(name='bn1')(h1)

    encoding = Dense(hidden[1], activation='relu', name='encoding')(h1)
    # bn2 = BatchNormalization(name='bn2')(encoding)

    h3 = Dense(hidden[2], activation='relu', name='h3')(encoding)
    # bn3 = BatchNormalization(name='bn3')(h3)

    h4 = Dense(hidden[3], activation='sigmoid', name='h4')(h3)

    model = Model(inputs=input_layer, outputs=h4)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()

    encoder = Model(inputs=input_layer, outputs=encoding)  # or bn2
    return model, encoder


def shared_autoencoder_models(h_unsw, h_nsl):
    unsw = Input(shape=(unsw_dim, ), name='input_unsw')
    nsl = Input(shape=(nsl_dim, ), name='input_nsl')

    h1_unsw = Dense(h_unsw[0], activation='relu', name='h1_unsw')(unsw)
    h1_nsl = Dense(h_nsl[0], activation='relu', name='h1_nsl')(nsl)
    # h1_unsw = BatchNormalization(name='bn1_unsw')(h1_unsw)
    # h1_nsl = BatchNormalization(name='bn1_nsl')(h1_nsl)

    shared_ae = Dense(h_unsw[1], activation='relu', name='shared')
    shared_unsw = shared_ae(h1_unsw)
    shared_nsl = shared_ae(h1_nsl)
    # bns_unsw = BatchNormalization(name='bn2_unsw')(shared_unsw)
    # bns_nsl = BatchNormalization(name='bn2_nsl')(shared_nsl)

    h3_unsw = Dense(h_unsw[2], activation='relu', name='h3_unsw')(shared_unsw)
    h3_nsl = Dense(h_nsl[2], activation='relu', name='h3_nsl')(shared_nsl)
    # h3_unsw = BatchNormalization(name='bn3_unsw')(h3_unsw)
    # h3_nsl = BatchNormalization(name='bn3_nsl')(h3_nsl)

    h4_unsw = Dense(h_unsw[3], activation='sigmoid', name='h4_unsw')(h3_unsw)
    h4_nsl = Dense(h_nsl[3], activation='sigmoid', name='h4_nsl')(h3_nsl)

    model_unsw = Model(inputs=unsw, output=h4_unsw)
    model_unsw.compile(optimizer='adam', loss='binary_crossentropy')
    model_unsw.summary()

    model_nsl = Model(inputs=nsl, output=h4_nsl)
    model_nsl.compile(optimizer='adam', loss='binary_crossentropy')
    model_nsl.summary()

    encoder_unsw = Model(inputs=unsw, outputs=shared_unsw)
    encoder_nsl = Model(inputs=nsl, outputs=shared_nsl)

    return model_unsw, model_nsl, encoder_unsw, encoder_nsl


def run_regressor(X, y, X_test, y_test, cls_hidden):
    classifier = classifier_model(X.shape[1], cls_hidden)
    classifier.fit(X, y, batch_size, num_epochs, verbose=1)
    scores = classifier.evaluate(X, y, X_test.shape[0], verbose=0)
    train_accu = scores[1]
    scores = classifier.evaluate(X_test, y_test, X_test.shape[0], verbose=0)
    test_accu = scores[1]
    return (train_accu, test_accu)


def run_single(X, X_test, y, y_test, hidden, cls_hidden):
    logger.info('Single AE %s with Classifier %s' % (hidden, cls_hidden))
    model, encoder = single_encoder_model(X.shape[1], hidden)
    model.fit(X, X, batch_size, num_epochs, verbose=1)
    EX = encoder.predict(X)
    EX_test = encoder.predict(X_test)

    accu = run_regressor(EX, y, EX_test, y_test, cls_hidden)
    return EX, EX_test, accu


def run_autoencoder(unsw_hidden, nsl_hidden, cls_hidden):
    EX_unsw, EX_unsw_test, accu = run_single(X_unsw, X_unsw_test,
                                             y_unsw, y_unsw_test,
                                             unsw_hidden, cls_hidden)
    logger.info('UNSW train accu %.6f' % accu[0])
    logger.info('UNSW test accu %.6f' % accu[1])
    ae['unsw']['train'] = accu[0]
    ae['unsw']['test'] = accu[1]

    EX_nsl, EX_nsl_test, accu = run_single(X_nsl, X_nsl_test,
                                           y_nsl, y_nsl_test,
                                           nsl_hidden, cls_hidden)
    logger.info('NSL train accu %.6f' % accu[0])
    logger.info('NSL test accu %.6f' % accu[1])
    ae['nsl']['train'] = accu[0]
    ae['nsl']['test'] = accu[1]

    UX = np.concatenate((EX_unsw, EX_nsl), axis=0)
    Uy = np.concatenate((y_unsw, y_nsl), axis=0)

    classifier = classifier_model(UX.shape[1], cls_hidden)
    classifier.fit(UX, Uy, batch_size, num_epochs, verbose=1)

    score = classifier.evaluate(EX_unsw, y_unsw, EX_unsw.shape[0], verbose=0)
    ae_unified['unsw']['train'].append(score[1])
    logger.info('Unified UNSW train accu %.6f' % score[1])
    score = classifier.evaluate(EX_unsw_test, y_unsw_test,
                                EX_unsw_test.shape[0], verbose=0)
    ae_unified['unsw']['test'].append(score[1])
    logger.info('Unified UNSW test accu %.6f' % score[1])

    score = classifier.evaluate(EX_nsl, y_nsl, EX_nsl.shape[0], verbose=0)
    ae_unified['nsl']['train'].append(score[1])
    logger.info('Unified NSL train accu %.6f' % score[1])
    score = classifier.evaluate(EX_nsl_test, y_nsl_test,
                                EX_nsl_test.shape[0], verbose=0)
    ae_unified['nsl']['test'].append(score[1])
    logger.info('Unified NSL test accu %.6f' % score[1])


def run_shared_autoencoder(unsw_hidden, nsl_hidden, cls_hidden):
    unsw_model, nsl_model, unsw_encoder, nsl_encoder = \
        shared_autoencoder_models(unsw_hidden, nsl_hidden)
    encoder_loss = [[], []]
    for _ in range(num_epochs):
        unsw_model.fit(X_unsw, X_unsw, batch_size, epochs=1)
        encoder_loss[0].append(
            unsw_model.evaluate(X_unsw, X_unsw, unsw_size, verbose=0))
        encoder_loss[1].append(
            nsl_model.evaluate(X_nsl, X_nsl, nsl_size, verbose=0))

        nsl_model.fit(X_nsl, X_nsl, batch_size, epochs=1)
        encoder_loss[0].append(
            unsw_model.evaluate(X_unsw, X_unsw, unsw_size, verbose=0))
        encoder_loss[1].append(
            nsl_model.evaluate(X_nsl, X_nsl, nsl_size, verbose=0))

    sae['unsw_loss'].append(encoder_loss[0])
    sae['nsl_loss'].append(encoder_loss[0])
    # Get the shared representation of both datasets
    EX_unsw = unsw_encoder.predict(X_unsw)
    EX_unsw_test = unsw_encoder.predict(X_unsw_test)
    accu = run_regressor(EX_unsw, y_unsw, EX_unsw_test, y_unsw_test, cls_hidden)
    sae['unsw']['train'] = accu[0]
    sae['unsw']['test'] = accu[1]

    EX_nsl = nsl_encoder.predict(X_nsl)
    EX_nsl_test = nsl_encoder.predict(X_nsl_test)
    accu = run_regressor(EX_nsl, y_nsl, EX_nsl_test, y_nsl_test, cls_hidden)
    sae['nsl']['train'] = accu[0]
    sae['nsl']['test'] = accu[1]

    UX = np.concatenate((EX_unsw, EX_nsl), axis=0)
    Uy = np.concatenate((y_unsw, y_nsl), axis=0)

    classifier = classifier_model(UX.shape[1], cls_hidden)
    classifier.fit(UX, Uy, batch_size, num_epochs, verbose=1)

    score = classifier.evaluate(EX_unsw, y_unsw, EX_unsw.shape[0], verbose=0)
    sae_unified['unsw']['train'].append(score[1])
    logger.info('Unified UNSW train accu %.6f' % score[1])
    score = classifier.evaluate(EX_unsw_test, y_unsw_test,
                                EX_unsw_test.shape[0], verbose=0)
    sae_unified['unsw']['test'].append(score[1])
    logger.info('Unified UNSW test accu %.6f' % score[1])

    score = classifier.evaluate(EX_nsl, y_nsl, EX_nsl.shape[0], verbose=0)
    sae_unified['nsl']['train'].append(score[1])
    logger.info('Unified NSL train accu %.6f' % score[1])
    score = classifier.evaluate(EX_nsl_test, y_nsl_test,
                                EX_nsl_test.shape[0], verbose=0)
    sae_unified['nsl']['test'].append(score[1])
    logger.info('Unified NSL test accu %.6f' % score[1])


def run_master(shared_size, cls_hidden):
    unsw_hidden = [256, shared_size, 256, unsw_dim]
    nsl_hidden = [256, shared_size, 256, nsl_dim]
    multicore_session()
    logger.info('Train %d epochs and %d batch' % (num_epochs, batch_size))
    run_autoencoder(unsw_hidden, nsl_hidden, cls_hidden)
    run_shared_autoencoder(unsw_hidden, nsl_hidden, cls_hidden)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('SharedAEX2')
    root = 'SharedAutoEncoder/'
    hdlr = logging.FileHandler(root + 'accuracy.log')
    formatter = logging.Formatter('%(asctime)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    X_unsw, X_unsw_test, y_unsw, y_unsw_test = get_unsw_data()
    X_nsl, X_nsl_test, y_nsl, y_nsl_test = get_nsl_data()
    (unsw_size, unsw_dim) = X_unsw.shape
    (nsl_size, nsl_dim) = X_nsl.shape

    mm = MinMaxScaler()
    mm.fit(X_unsw)
    X_unsw = mm.transform(X_unsw)
    X_unsw_test = mm.transform(X_unsw_test)

    mm = MinMaxScaler()
    mm.fit(X_nsl)
    X_nsl = mm.transform(X_nsl)
    X_nsl_test = mm.transform(X_nsl_test)

    # layer_sizes = [128, 240, 320, 400]
    layer_sizes = [480]
    h_cls = 400
    mult = 2
    num_epochs = 20
    num_runs = 10
    batch_size = 100
    beta = 0.00

    for hs in layer_sizes:
        ae = {'unsw': {'train': [], 'test': []},
              'unsw_loss': [], 'nsl_loss': [],
              'nsl': {'train': [], 'test': []}}
        ae_unified = {'unsw': {'train': [], 'test': []},
                      'unsw_loss': [], 'nsl_loss': [],
                      'nsl': {'train': [], 'test': []}}
        sae = {'unsw': {'train': [], 'test': []},
               'unsw_loss': [], 'nsl_loss': [],
               'nsl': {'train': [], 'test': []}}
        sae_unified = {'unsw': {'train': [], 'test': []},
                       'unsw_loss': [], 'nsl_loss': [],
                       'nsl': {'train': [], 'test': []}}

        logger.info('***************************************************')
        logger.info('*******  Start %d runs with shared layer %d  *******'
                    % (num_runs, hs))
        logger.info('***************************************************')
        for i in range(num_runs):
            logger.info('*** Run index %d ***' % i)
            run_master(hs, h_cls)

        result = {'ae': ae, 'sae': sae,
                  'ae_unified': ae_unified, 'sae_unified': sae_unified}
        pprint(result)
        output = open(root + 'result_runs%d_U%d.pkl' % (num_runs, hs), 'wb+')
        pickle.dump(result, output)
        output.close()
