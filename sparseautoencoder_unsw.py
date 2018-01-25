from __future__ import print_function, division
import numpy as np
from netlearner.utils import min_max_scale, hyperparameter_summary
from netlearner.utils import permutate_dataset, measure_prediction
from netlearner.autoencoder import SparseAutoencoder
from preprocess import unsw
import tensorflow as tf
from math import ceil
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout

np.random.seed(4567)
tf.set_random_seed(4567)
model_dir = 'SparseAE/'
unsw.generate_dataset(True, model_dir)
data_dir = model_dir + 'UNSW/'
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

pretrain = True
num_epoch = 2
if pretrain is True:
    feature_size = train_dataset.shape[1]
    encoder_size = 800
    init_lr = 0.01
    batch_size = 100
    num_steps = ceil(train_dataset.shape[0] / batch_size * num_epoch)
    sae = SparseAutoencoder(feature_size, encoder_size, data_dir,
                            optimizer=tf.train.AdamOptimizer,
                            sparsity=0.25, sparsity_weight=0.1,
                            init_lr=init_lr, decay_steps=num_steps)
    sae.train_with_labels(train_dataset, train_labels, batch_size,
                          int(num_steps), valid_dataset)
    test_loss = sae.calc_reconstruct_loss(test_dataset)
    print("Testset reconstruction loss: %f" % test_loss)
    sae_w = sae.get_encode_weights()
    sae_b = sae.get_encode_biases()
    hyperparameter = {'encoding units': encoder_size,
                      'init_lr': init_lr,
                      'num_epoch': num_epoch,
                      'num_steps': num_steps,
                      'act_func': 'sigmoid',
                      'batch_size': batch_size, }
    hyperparameter_summary(sae.dirname, hyperparameter)

    sae_train_dataset = sae.encode_dataset(train_dataset)
    sae_valid_dataset = sae.encode_dataset(valid_dataset)
    sae_test_dataset = sae.encode_dataset(test_dataset)
    tf.reset_default_graph()

    input_layer = Input(shape=(train_dataset.shape[1], ), name='input')
    h1 = Dense(encoder_size, activation='sigmoid', name='h1')(input_layer)
    h1 = Dropout(0.8)(h1)
    h2 = Dense(480, activation='sigmoid', name='h2')(h1)
    sm = Dense(2, activation='softmax', name='output')(h2)
    mlp = Model(inputs=input_layer, outputs=sm, name='rbm_mlp')
    mlp.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'])
    mlp.summary()
    mlp.get_layer('h1').set_weights([sae_w, sae_b])
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
