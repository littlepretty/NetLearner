
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# In[2]:

raw_train_dataset, train_labels = np.load('NSL-KDD/train_dataset.npy'), np.load('NSL-KDD/train_ref.npy')
raw_valid_dataset, valid_labels = np.load('NSL-KDD/valid_dataset.npy'), np.load('NSL-KDD/valid_ref.npy')
raw_test_dataset, test_labels = np.load('NSL-KDD/test_dataset.npy'), np.load('NSL-KDD/test_ref.npy')
print('Training set', raw_train_dataset.shape, train_labels.shape)
print('Validation set', raw_valid_dataset.shape, valid_labels.shape)
print('Test set', raw_test_dataset.shape, test_labels.shape)


# In[3]:

# feature_mins = raw_train_dataset.min(axis=0)
# feature_maxs = raw_train_dataset.max(axis=0)
# print(feature_mins.shape)
# print(feature_maxs.shape)
    
# def min_max_normalize(dataset):
#     normalized_dataset = np.zeros(dataset.shape)
#     for (i, row) in enumerate(dataset):
#         normalized_dataset[i, :] = np.divide(row - feature_mins, feature_maxs - feature_mins)

#     return normalized_dataset
    
# train_dataset = min_max_normalize(raw_train_dataset)
# valid_dataset = min_max_normalize(raw_valid_dataset)
# test_dataset = min_max_normalize(raw_test_dataset)

def standard_scale(X_train, X_valid, X_test):
    preprocessor = StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)
    X_test = preprocessor.transform(X_test)
    return X_train, X_valid, X_test

train_dataset, valid_dataset, test_dataset = standard_scale(raw_train_dataset,
                                                            raw_valid_dataset,
                                                            raw_test_dataset)

# print(train_dataset.min(axis=0))
# print(train_dataset.max(axis=0))
print(np.mean(train_dataset, axis=0))
print(np.std(train_dataset, axis=0))

train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[4]:

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) ==
                           np.argmax(labels, 1)) / predictions.shape[0])


def compute_classification_table(predictions, labels):
    num_classes = labels.shape[1]
    class_table = np.zeros((num_classes, num_classes))
    predicted_class = np.argmax(predictions, 1)
    actual_class = np.argmax(labels, 1)
    for (a, p) in zip(actual_class, predicted_class):
        class_table[a][p] += 1

    return class_table


def correct_percentage(matrix):
    epsilon = 1e-20
    num_classes = matrix.shape[0]
    act2pred = [matrix[i][i] / (np.sum(matrix[i, :]) + epsilon) for i in range(num_classes)]
    pred2act = [matrix[i][i] / (np.sum(matrix[:, i]) + epsilon) for i in range(num_classes)]
    print(act2pred)
    print(pred2act)


# In[5]:

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)


# In[23]:

class Autoencoder(object):
    def __init__(self, feature_size, encode_size, encode_lr=0.01, l2reg=0.003,
                 transfer_func=tf.nn.softplus, optimizer = tf.train.GradientDescentOptimizer):
        self.feature_size = feature_size
        self.encode_size = encode_size
        self.l2reg = l2reg
        self.transfer_func = transfer_func

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.feature_size])

        self.encode = self.transfer_func(tf.add(tf.matmul(self.x,
                                                          self.weights['w1']),
                                                self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.encode,
                                               self.weights['w2']),
                                     self.weights['b2'])

        self.regterm = self._create_regterm()

        self.mean_squared_loss = 0.5 * tf.reduce_mean(tf.pow(
                tf.sub(self.reconstruction, self.x), 2.0))

        self.cross_entropy_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.reconstruction, self.x))
                
        self.loss = tf.add(self.mean_squared_loss, self.regterm)
        self.optimizer = optimizer(encode_lr).minimize(self.mean_squared_loss)

        init = tf.initialize_all_variables()
        
        self.sess = tf.Session()
        self.sess.run(init)
        print('Autoencoder built and initialized')

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.feature_size,
                                                    self.encode_size))
        all_weights['b1'] = tf.Variable(tf.zeros([self.encode_size],
                                                 dtype=tf.float32))
        
        all_weights['w2'] = tf.Variable(tf.zeros([self.encode_size, self.feature_size],
                                                 dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.feature_size],
                                                 dtype=tf.float32))
        
        return all_weights
    
    def _create_regterm(self, reg_func=tf.nn.l2_loss):
        self.regterm = reg_func([0.0])
        for name, param in self.weights.items():
            self.regterm = tf.add(self.regterm, reg_func(param))
        return tf.mul(self.regterm, self.l2reg)

    def partial_fit(self, X, loss_name='mean_squared'):
        opt, loss, reg = self.sess.run([self.optimizer, self.mean_squared_loss, self.regterm],
                                  feed_dict={self.x: X})
        return loss, reg

    def calc_total_loss(self, X):
        return self.sess.run(self.mean_squared_loss, feed_dict={self.x: X})

    def encodeDataset(self, X):
        return self.sess.run(self.encode, feed_dict={self.x: X})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction,
                             feed_dict={self.x: X})

    def getEncodeWeights(self):
        return self.sess.run(self.weights['w1'])

    def getEncodeBiases(self):
        return self.sess.run(self.weights['b1'])
    
    def train(self, train_dataset, batch_size, num_epochs):
        num_steps = int(num_samples * num_epochs / batch_size)
        display_step = num_steps / 10
        print('Training for %d steps' % num_steps)

        for step in range(num_steps):
            offset = (batch_size * step) % (num_samples - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]

            loss, reg = self.partial_fit(batch_data)
            if step % display_step == 0:
                print("Minibatch loss at step %d:\t%f(regterm=%f)" % (step, loss, reg))

        print('Autoencoder trained')
        train_loss = autoencoder.calc_total_loss(train_dataset)
        print("Trainset decode loss: %f" % train_loss)


# In[24]:

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]
encoder_size = 64
encoder_lr = 0.1
beta=0.009

autoencoder = Autoencoder(feature_size, encoder_size, encoder_lr, beta)
batch_size = 20000
num_epochs = 600.0
autoencoder.train(train_dataset, batch_size, num_epochs)
test_loss = autoencoder.calc_total_loss(test_dataset)
print("Testset decode loss: %f" % test_loss)


# In[25]:

encoded_train_dataset = autoencoder.encodeDataset(train_dataset)
encoded_valid_dataset = autoencoder.encodeDataset(valid_dataset)
encoded_test_dataset = autoencoder.encodeDataset(test_dataset)
print('Dataset encoded')


# In[26]:

init_learning_rate = 0.99
decay_steps = 100000

decay_base = 0.96
num_hidden_classify = 256
beta = 0.001

graph_class = tf.Graph()
batch_size = 100000

with graph_class.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, [batch_size, encoder_size])
    tf_train_labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    tf_valid_dataset = tf.placeholder(tf.float32, [None, encoder_size])
    tf_test_dataset = tf.placeholder(tf.float32, [None, encoder_size])
    
    tf_all_train_dataset = tf.placeholder(tf.float32, [None, encoder_size])
    keep_prob = tf.placeholder(tf.float32)
    
#     W1_class = tf.Variable(tf.truncated_normal(
#             shape=[encoder_layer, num_hidden_classify], stddev=0.12))
#     W2_class = tf.Variable(tf.truncated_normal(
#             shape=[num_hidden_classify, num_labels], stddev=0.08))
    W1_class = tf.Variable(xavier_init(encoder_size, num_hidden_classify))
    W2_class = tf.Variable(xavier_init(num_hidden_classify, num_labels))
    
    b1_class = tf.Variable(tf.zeros(
            shape=[num_hidden_classify], dtype=tf.float32))
    b2_class = tf.Variable(tf.zeros(
            shape=[num_labels], dtype=tf.float32))
    
    def forward(dataset):
        logits = tf.matmul(dataset, W1_class) + b1_class
        activity = tf.nn.relu(logits)
        dropout = tf.nn.dropout(activity, keep_prob)
        model = tf.matmul(dropout, W2_class) + b2_class
        return model;
    
    logits = forward(tf_train_dataset)
    class_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    regterm_class = tf.add(tf.nn.l2_loss(W1_class), tf.nn.l2_loss(b1_class))
    regterm_class= tf.add(regterm_class, tf.nn.l2_loss(W2_class))
    regterm_class= tf.add(regterm_class, tf.nn.l2_loss(b2_class))
    class_loss = class_loss + beta * regterm_class
    
    # exponentially decaying learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               decay_steps, decay_base,
                                               staircase=True)
    # notice that here optimizer's minimize function will help us increment global step
    optimizer_class = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(class_loss, global_step=global_step)

    train_predict = tf.nn.softmax(logits)
    valid_predict = tf.nn.softmax(forward(tf_valid_dataset))
    test_predict = tf.nn.softmax(forward(tf_test_dataset))
    all_train_predict = tf.nn.softmax(forward(tf_all_train_dataset))
    
    print('Softmax classification built')


# In[39]:

with tf.Session(graph=graph_class) as classify_session:
    tf.initialize_all_variables().run()
    print('Initialized')
    num_steps = 1301
    for step in range(num_steps):
        offset = (batch_size * step) % (train_labels.shape[0] - batch_size)
        batch_data = encoded_train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels,
                     keep_prob: 0.8}
        _, l, train_predictions = classify_session.run([optimizer_class, class_loss, train_predict],
                                              feed_dict=feed_dict)
        if step % (num_steps / 10) == 0 or step == num_steps - 1:
            print("Bigbatch loss at step %d: \t%f" % (step, l))
            accu = accuracy(train_predictions, batch_labels);
            print("Bigbatch train accuracy: \t\t%f%%" % accu)
            if accu >= 99.0:
                break;
            # feed_dict = {tf_valid_dataset: encoded_valid_dataset, keep_prob: 1.0}
            # print("Minibatch validation accuracy: \t\t\t\t%f%%" %
            # accuracy(valid_predict.eval(feed_dict=feed_dict), valid_labels))
    
    # feed_dict_train = {tf_all_train_dataset: encoded_train_dataset,
    #                    keep_prob: 1.0}
    # all_train_predictions = all_train_predict.eval(feed_dict=feed_dict_train)

    # feed_dict_valid = {tf_valid_dataset: encoded_valid_dataset,
    #                   keep_prob: 1.0}
    # valid_predictions = valid_predict.eval(feed_dict=feed_dict_valid)


    print('Classifier trained')
    
    feed_dict_test = {tf_test_dataset: encoded_test_dataset,
                      keep_prob: 1.0}
    test_predictions = test_predict.eval(feed_dict=feed_dict_test)
    print('Test dataset classified')


# In[40]:

def measure_prediction(predictions, labels, dataset_name='Test'):
    print("%sset accuracy: %f%%" % (dataset_name, accuracy(predictions, labels)))
    headers = [str(i) for i in range(labels.shape[1])]
    class_table = compute_classification_table(predictions, labels)
    print(tabulate(class_table, headers))
    correct_percentage(class_table)
    
# measure_prediction(all_train_predictions, train_labels, 'Train')
# measure_prediction(valid_predictions, valid_labels, 'Valid')
measure_prediction(test_predictions, test_labels, 'Test')


# In[ ]:



