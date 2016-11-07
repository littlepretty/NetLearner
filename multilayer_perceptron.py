
# coding: utf-8

# In[8]:

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tabulate import tabulate


raw_train_dataset = np.load('NSL-KDD/train_dataset.npy')
train_labels = np.load('NSL-KDD/train_ref.npy')
raw_valid_dataset = np.load('NSL-KDD/valid_dataset.npy')
valid_labels = np.load('NSL-KDD/valid_ref.npy')
raw_test_dataset = np.load('NSL-KDD/test_dataset.npy')
test_labels = np.load('NSL-KDD/test_ref.npy')
print('Training set', raw_train_dataset.shape, train_labels.shape)
print('Validation set', raw_valid_dataset.shape, valid_labels.shape)
print('Test set', raw_test_dataset.shape, test_labels.shape)
print(raw_train_dataset.min(axis=0))
print(raw_train_dataset.max(axis=0))


# Mean normalize data
feature_means = np.mean(raw_train_dataset, axis=0)
feature_stds = np.std(raw_train_dataset, axis=0)
print(feature_means.shape)
print(feature_stds.shape)


def mean_normalize(dataset):
    normalized_dataset = np.zeros(dataset.shape)
    for (i, row) in enumerate(dataset):
        normalized_dataset[i, :] = np.divide(row - feature_means, feature_stds)

    return normalized_dataset

train_dataset = mean_normalize(raw_train_dataset)
valid_dataset = mean_normalize(raw_valid_dataset)
test_dataset = mean_normalize(raw_test_dataset)

print(np.mean(train_dataset, axis=0))
print(np.std(train_dataset, axis=0))

train_dataset = np.concatenate((train_dataset, valid_dataset), axis=0)
train_labels = np.concatenate((train_labels, valid_labels), axis=0)


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
    act2pred = [matrix[i][i] / (np.sum(matrix[i, :]) + epsilon)
                for i in range(num_classes)]
    pred2act = [matrix[i][i] / (np.sum(matrix[:, i]) + epsilon)
                for i in range(num_classes)]
    print(act2pred)
    print(pred2act)


def xavier_init(fan_in, fan_out, constant=1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


# ### Perceptron with 3 hidden layers

# In[43]:

class MultilayerPerceptron(object):
    def __init__(self, feature_size, hidden_layer_sizes, num_labels,
                 init_learning_rate=0.99, decay_steps=100000,
                 decay_base=0.96, trans_func=tf.nn.relu,
                 reg_func=tf.nn.l2_loss, beta=0.009):
        self.feature_size = feature_size
        self.layer_sizes = hidden_layer_sizes + [num_labels]
        self.num_labels = num_labels
        self.trans_func = trans_func

        self.params = self._create_variables()

        self.x = tf.placeholder(tf.float32, [None, feature_size])
        self.t = tf.placeholder(tf.float32, [None, num_labels])
        self.keep_prob = tf.placeholder(tf.float32)

        self.final_logits = self._create_forward()
        self.classify_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.final_logits, self.t))
        self.regterm = self._create_regterm(reg_func, beta)

        self.loss = self.classify_loss + self.regterm

        # exponentially decaying learning rate
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            init_learning_rate, global_step, decay_steps,
            decay_base, staircase=True)
        self.optimizer = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss, global_step=global_step)

        self.predict = tf.nn.softmax(self.final_logits)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
        print('Multilayer Perceptron build and initialized')

    def _create_variables(self):
        fsize = self.feature_size
        weights = dict()
        for (i, hsize) in enumerate(self.layer_sizes):
            weights['w%d' % i] = tf.Variable(xavier_init(fsize, hsize))
            weights['b%d' % i] = tf.Variable(tf.zeros(shape=[hsize],
                                                      dtype=tf.float32))
            fsize = hsize

        return weights

    def _create_forward(self):
        next_input = self.x
        for (i, hsize) in enumerate(self.layer_sizes):
            logits = tf.add(tf.matmul(next_input, self.params['w%d' % i]),
                            self.params['b%d' % i])
            activity = self.trans_func(logits)
            next_input = tf.nn.dropout(activity, self.keep_prob)
        return next_input

    def _create_regterm(self, reg_func, beta):
        regterm = reg_func([0.0])
        for name, var in self.params.items():
            regterm = tf.add(regterm, reg_func(var))
        return tf.mul(beta, regterm)

    def fit(self, X, T, prob):
        opt, loss, reg = self.sess.run(
            [self.optimizer, self.loss, self.regterm],
            feed_dict={self.x: X, self.t: T, self.keep_prob: prob})
        return loss, reg

    def makePrediction(self, X, prob=1.0):
        return self.sess.run(self.predict,
                             feed_dict={self.x: X, self.keep_prob: prob})

    def calc_total_loss(self, X, T, prob=1.0):
        return self.sess.run(self.loss,
                             feed_dict={self.x: X, self.t: T,
                                        self.keep_prob: prob})

    def train(self, train_dataset, train_labels, batch_size, num_samples,
              num_epochs=10.0, keep_prob=0.5):
        num_steps = int(num_samples * num_epochs / batch_size)
        display_step = num_steps / 10
        print('Training for %d steps' % num_steps)

        for step in range(num_steps):
            offset = (batch_size * step) % (num_samples - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            loss, reg = self.fit(batch_data, batch_labels, keep_prob)
            if step % display_step == 0:
                print("Minibatch loss at step %d:\t%f(regterm=%f)"
                      % (step, loss, reg))
                batch_predict = self.makePrediction(batch_data)
                print("Minibatch train accuracy: %f%%" %
                      accuracy(batch_predict, batch_labels))

        print('Multilayer Perceptron trained')
        train_loss = self.calc_total_loss(train_dataset, train_labels)
        train_predict = self.makePrediction(train_dataset)
        train_accuracy = accuracy(train_predict, train_labels)
        print("Trainset total loss: %f" % train_loss)
        print("Trainset total accuracy: %f" % train_accuracy)


# ### Train with very large batch size

# In[47]:

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]
hidden_layer_sizes = [512, 256, 64]

mp_classifier = MultilayerPerceptron(feature_size, hidden_layer_sizes,
                                     num_labels, trans_func=tf.nn.relu)
batch_size = 40000
num_epochs = 500.0
mp_classifier.train(train_dataset, train_labels, batch_size,
                    num_samples, num_epochs)
test_predict = mp_classifier.makePrediction(test_dataset)
test_accuracy = accuracy(test_predict, test_labels)
print("Testset total accuracy: %f" % test_accuracy)


# In[51]:

def measure_prediction(predictions, labels, dataset_name='Test'):
    accu = accuracy(predictions, labels)
    print("%sset accuracy: %f%%" % (dataset_name, accu))
    headers = [str(i) for i in range(labels.shape[1])]
    class_table = compute_classification_table(predictions, labels)
    print(tabulate(class_table, headers))
    correct_percentage(class_table)

measure_prediction(test_predict, test_labels, 'Test')
