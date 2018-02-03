from __future__ import print_function
import numpy as np
from preprocess.unsw import generate_dataset
from netlearner.utils import measure_prediction, min_max_scale
from netlearner.utils import permutate_dataset

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
model_dir = 'KerasMLP/'
generate_dataset(True, model_dir)
data_dir = model_dir + 'UNSW/'
mlp_path = data_dir + 'mlp.h5'

raw_train_dataset = np.load(data_dir + 'train_dataset.npy')
raw_valid_dataset = np.load(data_dir + 'valid_dataset.npy')
raw_test_dataset = np.load(data_dir + 'test_dataset.npy')
train_labels = np.load(data_dir + 'train_labels.npy')
valid_labels = np.load(data_dir + 'valid_labels.npy')
test_labels = np.load(data_dir + 'test_labels.npy')
[train_dataset, valid_dataset, test_dataset] = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_dataset, train_labels = permutate_dataset(train_dataset, train_labels)
valid_dataset, valid_labels = permutate_dataset(valid_dataset, valid_labels)
test_dataset, test_labels = permutate_dataset(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 80
keep_prob = 0.8
num_epoch = 200
tail = 160
incremental = True
if incremental is False:
    num_samples, num_classes = train_labels.shape
    feature_size = train_dataset.shape[1]
    hidden_size = [800, 480]
    input_layer = Input(shape=(feature_size, ), name='input')
    h1 = Dense(hidden_size[0], activation='relu', name='h1')(input_layer)
    h1 = Dropout(keep_prob)(h1)
    h2 = Dense(hidden_size[1], activation='relu', name='h2')(h1)
    sm = Dense(num_classes, activation='softmax', name='output')(h2)
    mlp = Model(inputs=input_layer, outputs=sm, name='sae_mlp')
    mlp.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    mlp.summary()
else:
    mlp = load_model(mlp_path)

weights = {0: 0.5, 1: 0.5}
hist = mlp.fit(train_dataset, train_labels, batch_size, num_epoch,
               verbose=1, class_weight=weights,
               validation_data=(test_dataset, test_labels))
score = mlp.evaluate(test_dataset, test_labels, test_dataset.shape[0])
print('%s = %s' % (mlp.metrics_names, score))

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
output = open(data_dir + 'Runs%d.pkl' % (num_epoch), 'wb')
pickle.dump(hist.history, output)
output.close()
mlp.save(mlp_path)
