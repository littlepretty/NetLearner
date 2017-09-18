from __future__ import print_function
import numpy as np
import tensorflow as tf
from netlearner.utils import min_max_scale, permutate_dataset, plot_samples
from netlearner.gan import GenerativeAdversarialNets
from netlearner.utils import hyperparameter_summary
from netlearner.multilayer_perceptron import MultilayerPerceptron
from math import ceil


def generate_fake_data(dataset, labels):
    num_samples, input_dim = dataset.shape
    _, num_labels = labels.shape
    noise_dim = 100
    batch_size = 100
    num_epochs = 180
    keep_prob = 0.9
    init_lr = 0.001
    num_steps = ceil(num_samples / batch_size * num_epochs)
    decay_steps = int(num_steps / 10)
    G_hidden_layer = 160
    D_hidden_layer = 160
    with tf.name_scope('GAN'):
        gan = GenerativeAdversarialNets(noise_dim, input_dim,
                                        G_hidden_layer, D_hidden_layer,
                                        init_lr, decay_steps,
                                        name="VanillaGAN-UNSW")
        gan.train(batch_size, dataset, int(num_steps), keep_prob)
        fake_data = gan.synthesize(num_samples)
        print(fake_data.shape)
        gan.close()
    return fake_data


def classify(train_dataset, train_labels, valid_dataset, valid_labels,
             test_dataset, test_labels):
    num_samples, feature_size = train_dataset.shape
    num_labels = train_labels.shape[1]
    batch_size = 80
    keep_prob = 0.80
    beta = 0.00001
    weights = [1.0, 10.0]
    num_epochs = [80]
    init_lrs = [0.001]
    hidden_layer_sizes = [
        [400],
        # [800, 640], [160, 80], [80, 40],
        # [400, 360, 320],
        # [160, 120, 80], [120, 80, 40],
    ]
    for hidden_layer_size in hidden_layer_sizes:
        for init_lr in init_lrs:
            for num_epoch in num_epochs:
                num_steps = int(train_dataset.shape[0] / batch_size * num_epoch)
                decay_steps = num_steps / num_epoch
                mp_classifier = MultilayerPerceptron(feature_size,
                                                     hidden_layer_size,
                                                     num_labels, init_lr,
                                                     decay_steps,
                                                     beta, tf.nn.relu,
                                                     tf.nn.l2_loss, weights,
                                                     tf.train.AdamOptimizer,
                                                     name='GAN-MLP-UNSW')
                mp_classifier.train_with_labels(train_dataset, train_labels,
                                                batch_size, num_steps,
                                                valid_dataset, valid_labels,
                                                test_dataset, test_labels,
                                                keep_prob)
                hyperparameter = {'hidden_layer_size': hidden_layer_size,
                                  'init_lr': init_lr,
                                  'num_epochs': num_epoch,
                                  'num_steps': num_steps,
                                  'regularization beta': beta,
                                  'optimizer': 'AdamOptimizer',
                                  'keep_prob': keep_prob,
                                  'act_func': 'RELU',
                                  'class_weights': weights,
                                  'batch_size': batch_size, }
                hyperparameter_summary(mp_classifier.dirname,
                                       hyperparameter)
                f = open(mp_classifier.dirname + '/test.log')
                print(f.read())
                f.close()
                mp_classifier.exit()


np.random.seed(1794)
raw_train_dataset = np.load('UNSW/train_dataset.npy')
train_labels = np.load('UNSW/train_labels.npy')
raw_valid_dataset = np.load('UNSW/valid_dataset.npy')
valid_labels = np.load('UNSW/valid_labels.npy')
raw_test_dataset = np.load('UNSW/test_dataset.npy')
test_labels = np.load('UNSW/test_labels.npy')

[train_dataset, valid_dataset, test_dataset] = min_max_scale(
    raw_train_dataset, raw_valid_dataset, raw_test_dataset)
train_dataset, train_labels = permutate_dataset(train_dataset, train_labels)
valid_dataset, valid_labels = permutate_dataset(valid_dataset, valid_labels)
test_dataset, test_labels = permutate_dataset(test_dataset, test_labels)

# Generate fake attacking data using GAN
attack_label = np.array([0.0, 1.0])  # attacking label = 1
attack_index = np.where(np.all(train_labels == attack_label, axis=1))[0]
attack_dataset = train_dataset[attack_index, :]
attack_labels = train_labels[attack_index, :]
print('Real attack set', attack_dataset.shape, attack_labels.shape)
# Plot what attacks look alike
sample_index = np.random.choice(attack_dataset.shape[0], 64, replace=False)
attack_samples = attack_dataset[sample_index, :]
plot_samples(attack_samples, 'UNSW', 0)

fake_dataset = generate_fake_data(attack_dataset, attack_labels)
mixed_dataset = np.concatenate((train_dataset, fake_dataset), axis=0)
mixed_labels = np.concatenate((train_labels, attack_labels), axis=0)
print('Mix trainset with fake attacks', mixed_dataset.shape,
      mixed_labels.shape)
tf.reset_default_graph()
classify(mixed_dataset, mixed_labels, valid_dataset, valid_labels,
         test_dataset, test_labels)
