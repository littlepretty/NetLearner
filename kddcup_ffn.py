
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tabulate import tabulate


# In[2]:

train_dataset, train_labels = np.load('KDDCup/train_dataset.npy'), np.load('KDDCup/train_ref.npy')
valid_dataset, valid_labels = np.load('KDDCup/valid_dataset.npy'), np.load('KDDCup/valid_ref.npy')
test_dataset, test_labels = np.load('KDDCup/test_dataset.npy'), np.load('KDDCup/test_ref.npy')
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)


# In[3]:

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


# In[4]:

batch_size = 1024
feature_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]

init_learning_rate = 0.99
decay_steps = 8000

decay_base = 0.96
hidden_units1 = 128
hidden_units2 = 512
hidden_units3 = 16
beta = 0.009
graph = tf.Graph()

with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, feature_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset, tf.float32)
    tf_test_dataset = tf.constant(test_dataset, tf.float32)

    tf_all_train_dataset = tf.placeholder(tf.float32, shape=train_dataset.shape)

    keep_prob = tf.placeholder(tf.float32, shape=None)

    W1 = tf.Variable(tf.truncated_normal([feature_size, hidden_units1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([hidden_units1]))
    W2 = tf.Variable(tf.truncated_normal([hidden_units1, hidden_units2], stddev=0.1))
    b2 = tf.Variable(tf.zeros([hidden_units2]))
    W3 = tf.Variable(tf.truncated_normal([hidden_units2, hidden_units3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([hidden_units3]))
    W4 = tf.Variable(tf.truncated_normal([hidden_units3, num_labels], stddev=0.1))
    b4 = tf.Variable(tf.zeros([num_labels]))

    def getThreeLayerNN(data_set):
        """Return a 3 layer logistic model"""
        l1 = tf.add(tf.matmul(data_set, W1), b1)
        hidden1 = tf.nn.relu(l1)
        dropout1 = tf.nn.dropout(hidden1, keep_prob)
        l2 = tf.matmul(dropout1, W2) + b2
        hidden2 = tf.nn.relu(l2)
        dropout2 = tf.nn.dropout(hidden2, keep_prob)
        l3 = tf.matmul(dropout2, W3) + b3
        hidden3 = tf.nn.relu(l3)
        dropout3 = tf.nn.dropout(hidden3, keep_prob)
        l4 = tf.matmul(dropout3, W4) + b4
        return l4

    model = getThreeLayerNN(tf_train_dataset)
    # regularizing loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model, tf_train_labels))
    use_regularization = True

    if use_regularization:
        reg_weights = tf.add(tf.nn.l2_loss(W1), tf.nn.l2_loss(W2))
        reg_weights = beta * tf.add(reg_weights, tf.nn.l2_loss(W3))
        loss = tf.add(loss, reg_weights)

    # exponentially decaying learning rate
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               decay_steps, decay_base,
                                               staircase=True)
    # notice that here optimizer's minimize function will help us increment global step
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_predict = tf.nn.softmax(model)
    valid_predict = tf.nn.softmax(getThreeLayerNN(tf_valid_dataset))
    test_predict = tf.nn.softmax(getThreeLayerNN(tf_test_dataset))

    all_train_predict = tf.nn.softmax(getThreeLayerNN(tf_all_train_dataset))

    print("3 Layer Model constructed")


# In[ ]:

num_steps = 6001
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (batch_size * step) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels,
                     keep_prob: 0.5}
        _, l, train_predictions = session.run([optimizer, loss, train_predict],
                                              feed_dict=feed_dict)
        if step % 1000 == 0:
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch train accuracy: %f%%" % accuracy(train_predictions, batch_labels))

            feed_dict = {tf_valid_dataset: valid_dataset,
                         keep_prob: 1.0}
            print("Minibatch validation accuracy: %f%%" %
                  accuracy(valid_predict.eval(feed_dict=feed_dict), valid_labels))

    feed_dict_test = {tf_test_dataset: test_dataset, keep_prob: 1.0}
    test_predictions = test_predict.eval(feed_dict=feed_dict_test)

    # feed_dict_train = {tf_all_train_dataset: train_dataset, keep_prob: 1.0}
    # all_train_predictions = all_train_predict.eval(feed_dict=feed_dict_train)

print('Training phase finished')


# In[34]:

def measure_prediction(predictions, labels, dataset_name='Test'):
    print("%sset accuracy: %f%%" % (dataset_name, accuracy(test_predictions, test_labels)))
    headers = [str(i) for i in range(labels.shape[1])]
    class_table = compute_classification_table(predictions, labels)
    print(tabulate(class_table, headers))
    correct_percentage(class_table)

# measure_prediction(all_train_predictions, train_labels)
measure_prediction(test_predictions, test_labels)
