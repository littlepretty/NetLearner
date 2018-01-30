from __future__ import print_function, division
import numpy as np
from netlearner.utils import min_max_scale, measure_prediction
from netlearner.utils import permutate_dataset
from netlearner.rbm import RestrictedBoltzmannMachine
from preprocess.nslkdd import generate_dataset
import tensorflow as tf
from math import ceil
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
import pickle
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
model_dir = 'RBM/'
generate_dataset(binary_label=False, one_hot_encoding=True, root=model_dir)
data_dir = model_dir + 'NSLKDD/'
mlp_path = data_dir + 'rbm_mlp.h5'

raw_train_dataset = np.load(data_dir + 'train_dataset.npy')
train_labels = np.load(data_dir + 'train_labels.npy')
raw_valid_dataset = np.load(data_dir + 'valid_dataset.npy')
valid_labels = np.load(data_dir + 'valid_labels.npy')
raw_test_dataset = np.load(data_dir + 'test_dataset.npy')
test_labels = np.load(data_dir + 'test_labels.npy')
[train_dataset, valid_dataset, test_dataset] = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_dataset, train_labels = permutate_dataset(train_dataset, train_labels)
valid_dataset, valid_labels = permutate_dataset(valid_dataset, valid_labels)
test_dataset, test_labels = permutate_dataset(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape)

incremental = True
if incremental is False:
    (num_samples, num_labels) = train_labels.shape
    feature_size = train_dataset.shape[1]
    num_hidden_rbm = 800
    rbm_lr = 0.01
    batch_size = 10
    num_epoch = 120
    num_steps = ceil(train_dataset.shape[0] / batch_size * num_epoch)
    rbm = RestrictedBoltzmannMachine(feature_size, num_hidden_rbm,
                                     batch_size, trans_func=tf.nn.sigmoid,
                                     num_labels=num_labels, dirname=data_dir)
    rbm.train_with_labels(train_dataset, train_labels, int(num_steps),
                          valid_dataset, rbm_lr)
    test_loss = rbm.calc_reconstruct_loss(test_dataset)
    print("Testset reconstruction error: %f" % test_loss)
    rbm_w = rbm.get_weights('w')
    rbm_b = rbm.get_weights('bh')
    """
    hrand = np.random.random((train_dataset.shape[0], num_hidden_rbm))
    rbm_train_dataset = rbm.encode_dataset(train_dataset, hrand)
    hrand = np.random.random((valid_dataset.shape[0], num_hidden_rbm))
    rbm_valid_dataset = rbm.encode_dataset(valid_dataset, hrand)
    hrand = np.random.random((test_dataset.shape[0], num_hidden_rbm))
    rbm_test_dataset = rbm.encode_dataset(test_dataset, hrand)
    """
    tf.reset_default_graph()
    input_layer = Input(shape=(train_dataset.shape[1], ), name='input')
    h1 = Dense(num_hidden_rbm, activation='sigmoid', name='h1')(input_layer)
    h1 = Dropout(0.8)(h1)
    h2 = Dense(480, activation='sigmoid', name='h2')(h1)
    sm = Dense(num_labels, activation='softmax', name='output')(h2)
    mlp = Model(inputs=input_layer, outputs=sm, name='rbm_mlp')
    mlp.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    mlp.summary()
    mlp.get_layer('h1').set_weights([rbm_w, rbm_b])
else:
    mlp = load_model(mlp_path)

num_epoch = 100
tail = 60
batch_size = 80
weights = {0: 0.05, 1: 0.15, 2: 0.05, 3: 0.6, 4: 0.15}
hist = mlp.fit(train_dataset, train_labels, batch_size, num_epoch,
               verbose=1, class_weight=weights,
               validation_data=(test_dataset, test_labels))
score = mlp.evaluate(test_dataset, test_labels, test_dataset.shape[0])
print('Final %s = %s' % (mlp.metrics_names, score))
avg_train = np.mean(hist.history['acc'][tail:])
avg_test = np.mean(hist.history['val_acc'][tail:])
std_train = np.std(hist.history['acc'][tail:])
std_test = np.std(hist.history['val_acc'][tail:])
print('Avg Train Accu: %.6f +/- %.6f' % (avg_train, std_train))
print('Avg Test Accu: %.6f +/ %.6f' % (avg_test, std_test))

predictions = mlp.predict(train_dataset)
measure_prediction(predictions, train_labels, data_dir, 'Train')
predictions = mlp.predict(test_dataset)
measure_prediction(predictions, test_labels, data_dir, 'Test')

output = open(data_dir + 'Runs%d.pkl' % num_epoch, 'wb')
pickle.dump(hist.history, output)
output.close()
mlp.save(mlp_path)
