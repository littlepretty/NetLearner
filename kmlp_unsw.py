from __future__ import print_function
import numpy as np
from preprocess.unsw import generate_dataset
from netlearner.utils import measure_prediction, min_max_scale
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout
import os
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
model_dir = 'KerasMLP/'
generate_dataset(False, True, model_dir)
data_dir = model_dir + 'UNSW/'
mlp_path = data_dir + 'mlp.h5'

train_dataset = np.load(data_dir + 'train_dataset.npy')
test_dataset = np.load(data_dir + 'test_dataset.npy')
train_labels = np.load(data_dir + 'train_labels.npy')
test_labels = np.load(data_dir + 'test_labels.npy')
train_dataset, _, test_dataset = min_max_scale(train_dataset,
                                               None, test_dataset)
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

batch_size = 40
keep_prob = 0.8
num_epoch = 240
tail = 200
incremental = False
if incremental is False:
    num_samples, num_classes = train_labels.shape
    feature_size = train_dataset.shape[1]
    hidden_size = [400, 256]
    input_layer = Input(shape=(feature_size, ), name='input')
    h1 = Dense(hidden_size[0], activation='tanh', name='h1')(input_layer)
    h1 = Dropout(keep_prob)(h1)
    h2 = Dense(hidden_size[1], activation='sigmoid', name='h2')(h1)
    sm = Dense(num_classes, activation='softmax', name='output')(h2)
    mlp = Model(inputs=input_layer, outputs=sm, name='mlp')
    mlp.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    mlp.summary()
else:
    mlp = load_model(mlp_path)

weights = None
"""
num_steps = 56000 / batch_size + 1
y = np.argmax(train_labels, axis=1)
X = np.array([train_dataset[y == i, :] for i in range(num_classes)])
Y = np.array([train_labels[y == i, :] for i in range(num_classes)])
even_data = np.zeros((num_steps * num_classes * batch_size, feature_size))
even_label = np.zeros((num_steps * num_classes * batch_size, num_classes))
for step in range(num_steps):
    start = step * num_classes * batch_size
    for i in range(num_classes):
        part, label = get_random_batch(X[i], Y[i], batch_size)
        index = range(start + i * batch_size, start + (i + 1) * batch_size)
        even_data[index, :] = part
        even_label[index, :] = label
"""

hist = mlp.fit(train_dataset, train_labels, batch_size, num_epoch,
               verbose=1, class_weight=weights, shuffle=False,
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
conf_table = measure_prediction(predictions, test_labels, data_dir, 'Test')
print(conf_table)
output = open(data_dir + 'Runs%d.pkl' % (num_epoch), 'wb')
pickle.dump(hist.history, output)
output.close()
mlp.save(mlp_path)
