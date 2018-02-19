from __future__ import print_function, division
import numpy as np
from netlearner.utils import min_max_scale, measure_prediction
from netlearner.utils import permutate_dataset
from preprocess.nslkdd import generate_dataset
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
# from keras import regularizers
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model_dir = 'SparseAE/'
generate_dataset(False, True, model_dir)
data_dir = model_dir + 'NSLKDD/'
mlp_path = data_dir + 'sae_mlp.h5'

raw_train_dataset = np.load(data_dir + 'train_dataset.npy')
raw_valid_dataset = np.load(data_dir + 'valid_dataset.npy')
raw_test_dataset = np.load(data_dir + 'test_dataset.npy')
train_labels = np.load(data_dir + 'train_labels.npy')
valid_labels = np.load(data_dir + 'valid_labels.npy')
test_labels = np.load(data_dir + 'test_labels.npy')
train_dataset, valid_dataset, test_dataset = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_dataset, train_labels = permutate_dataset(train_dataset, train_labels)
valid_dataset, valid_labels = permutate_dataset(valid_dataset, valid_labels)
test_dataset, test_labels = permutate_dataset(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape)

incremental = True
if incremental is False:
    num_epoch = 120
    batch_size = 80
    num_samples, num_classes = train_labels.shape
    feature_size = train_dataset.shape[1]
    encoder_size = 800

    X = Input(shape=(feature_size, ), name='input')
    # add a Dense layer with a L1 activity regularizer
    encoded = Dense(encoder_size, activation='relu', name='encoder')(X)
    decoded = Dense(feature_size, activation='sigmoid')(encoded)
    sae = Model(X, decoded)
    sae.compile(optimizer='adadelta', loss='binary_crossentropy')
    sae.summary()
    hist = sae.fit(train_dataset, train_dataset, batch_size, num_epoch,
                   verbose=1, validation_data=(test_dataset, test_dataset))
    test_loss = sae.evaluate(test_dataset, test_dataset)
    print("Testset reconstruction loss: %f" % test_loss)

    h1 = Dropout(0.8)(encoded)
    h2 = Dense(480, activation='relu', name='h2')(h1)
    sm = Dense(num_classes, activation='softmax', name='output')(h2)
    mlp = Model(inputs=X, outputs=sm, name='sae_mlp')
    mlp.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    mlp.summary()
else:
    mlp = load_model(mlp_path)

num_epoch = 100
tail = 60
batch_size = 90
weights = {0: 0.05, 1: 0.15, 2: 0.05, 3: 0.6, 4: 0.15}
hist = mlp.fit(train_dataset, train_labels,
               batch_size, epochs=num_epoch,
               verbose=1, class_weight=weights,
               validation_data=(test_dataset, test_labels))
score = mlp.evaluate(test_dataset, test_labels, test_dataset.shape[0])
print('Final %s = %s' % (mlp.metrics_names, score))
"""Average for the last runs"""
avg_train = np.mean(hist.history['acc'][tail:])
avg_test = np.mean(hist.history['val_acc'][tail:])
std_train = np.std(hist.history['acc'][tail:])
std_test = np.std(hist.history['val_acc'][tail:])
print('Avg Train Accu: %.6f +/- %.6f' % (avg_train, std_train))
print('Avg Test Accu: %.6f +/ %.6f' % (avg_test, std_test))
"""Confusion table"""
predictions = mlp.predict(train_dataset)
measure_prediction(predictions, train_labels, data_dir, 'Train')
predictions = mlp.predict(test_dataset)
measure_prediction(predictions, test_labels, data_dir, 'Test')

output = open(data_dir + 'Runs%d.pkl' % (num_epoch), 'wb')
pickle.dump(hist.history, output)
output.close()
mlp.save(mlp_path)
