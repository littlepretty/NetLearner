from __future__ import print_function, division
import numpy as np
from netlearner.utils import min_max_scale, measure_prediction
from netlearner.utils import hyperparameter_summary
from netlearner.rbm import RestrictedBoltzmannMachine
from preprocess.unsw import generate_dataset
import tensorflow as tf
from math import ceil
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout

tf.set_random_seed(9876)
model_dir = 'RBM/'
generate_dataset(True, model_dir)
data_dir = model_dir + 'UNSW/'
mlp_path = data_dir + 'rbm_mlp.h5'

raw_train_dataset = np.load(data_dir + 'train_dataset.npy')
train_labels = np.load(data_dir + 'train_labels.npy')
raw_valid_dataset = np.load(data_dir + 'valid_dataset.npy')
valid_labels = np.load(data_dir + 'valid_labels.npy')
raw_test_dataset = np.load(data_dir + 'test_dataset.npy')
test_labels = np.load(data_dir + 'test_labels.npy')
[train_dataset, valid_dataset, test_dataset] = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape)

pretrain = True
num_epoch = 120
if pretrain is True:
    num_samples = train_dataset.shape[0]
    feature_size = train_dataset.shape[1]
    num_hidden_rbm = 400
    rbm_lr = 0.01
    batch_size = 10
    num_steps = ceil(train_dataset.shape[0] / batch_size * num_epoch)
    rbm = RestrictedBoltzmannMachine(feature_size, num_hidden_rbm,
                                     batch_size, trans_func=tf.nn.sigmoid,
                                     num_labels=2,
                                     dirname=data_dir)
    rbm.train_with_labels(train_dataset, train_labels,
                          int(num_steps),
                          valid_dataset, rbm_lr)
    test_loss = rbm.calc_reconstruct_loss(test_dataset)
    print("Testset reconstruction error: %f" % test_loss)
    hyperparameter = {'#hidden units': num_hidden_rbm,
                      'init_lr': rbm_lr,
                      'num_epoch': num_epoch,
                      'num_steps': num_steps,
                      'act_func': 'sigmoid',
                      'batch_size': batch_size, }
    hyperparameter_summary(rbm.dirname, hyperparameter)
    rbm_w = rbm.get_weights('w')
    rbm_b = rbm.get_weights('bh')

    tf.reset_default_graph()
    input_layer = Input(shape=(train_dataset.shape[1], ), name='input')
    h1 = Dense(num_hidden_rbm, activation='sigmoid', name='h1')(input_layer)
    h1 = Dropout(0.8)(h1)
    h2 = Dense(480, activation='sigmoid', name='h2')(h1)
    sm = Dense(2, activation='softmax', name='output')(h2)
    mlp = Model(inputs=input_layer, outputs=sm, name='rbm_mlp')
    mlp.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    mlp.summary()
    mlp.get_layer('h1').set_weights([rbm_w, rbm_b])
else:
    mlp = load_model(mlp_path)

hist = mlp.fit(train_dataset, train_labels,
               batch_size=80, epochs=num_epoch, verbose=1)
score = mlp.evaluate(test_dataset, test_labels, test_dataset.shape[0])
print('%s = %s' % (mlp.metrics_names, score))
predictions = mlp.predict(train_dataset)
measure_prediction(predictions, train_labels, data_dir, 'Train')
predictions = mlp.predict(test_dataset)
measure_prediction(predictions, test_labels, data_dir, 'Test')
mlp.save(mlp_path)
