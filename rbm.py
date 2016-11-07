
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler


# In[2]:

raw_train_dataset, train_labels = np.load('NSL-KDD/train_dataset.npy'), np.load('NSL-KDD/train_ref.npy')
raw_valid_dataset, valid_labels = np.load('NSL-KDD/valid_dataset.npy'), np.load('NSL-KDD/valid_ref.npy')
raw_test_dataset, test_labels = np.load('NSL-KDD/test_dataset.npy'), np.load('NSL-KDD/test_ref.npy')
print('Training set', raw_train_dataset.shape, train_labels.shape)
print('Validation set', raw_valid_dataset.shape, valid_labels.shape)
print('Test set', raw_test_dataset.shape, test_labels.shape)


# In[3]:

feature_mins = raw_train_dataset.min(axis=0)
feature_maxs = raw_train_dataset.max(axis=0)
print(feature_mins.shape)
print(feature_maxs.shape)
    
def min_max_normalize(dataset):
    normalized_dataset = np.zeros(dataset.shape)
    for (i, row) in enumerate(dataset):
        normalized_dataset[i, :] = np.divide(row - feature_mins, feature_maxs - feature_mins)

    return normalized_dataset
    
train_dataset = min_max_normalize(raw_train_dataset)
valid_dataset = min_max_normalize(raw_valid_dataset)
test_dataset = min_max_normalize(raw_test_dataset)

print(train_dataset.min(axis=0))
print(train_dataset.max(axis=0))

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


# ### Objected RBM

# In[23]:

def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)

def sample_prob_dist(prob, rand):
    return tf.nn.relu(tf.sign(prob - rand))
    
class RestrictedBoltzmannMachine(object):
    def __init__(self, num_visible, num_hidden, batch_size,
                 lr=0.01, trans_func=tf.nn.sigmoid):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.lr = lr
        self.trans_func = trans_func
        
        self.weights = self._initialize_weights()
        
        # v is used for both cd and generate hidden states
        self.v = tf.placeholder(tf.float32, [None, num_visible])

        # h is just used for generate visible states
        self.h = tf.placeholder(tf.float32, [None, num_hidden])

        self.vrand = tf.placeholder(tf.float32, [None, num_visible])
        self.hrand = tf.placeholder(tf.float32, [None, num_hidden])
        
        self.batch_size = batch_size

        # generate hidden state from visible state x and random distribution hrand
        self.encode = self.sample_hidden_from_visible(self.v)[0]
        
        # generate visible state from hidden state x and random distribution vrand
        self.reconstruct = self.sample_visible_from_hidden(self.h)
        
        # training algorithm: constractive divergence
        self.updates = self._create_cd1()
        self.loss = self.updates[0]

        # measure the goodness of the weights
        self.goodness = self._create_goodness()
        
        init = tf.initialize_all_variables()
        
        self.sess = tf.Session()
        self.sess.run(init)
        print('Initialized')

    def _initialize_weights(self):
        weights = dict();
        weights['w'] = tf.Variable(xavier_init(self.num_visible, self.num_hidden))
        # weights['bh'] = tf.Variable(tf.zeros(shape=[self.num_hidden], dtype=tf.float32))
        # weights['bv'] = tf.Variable(tf.zeros(shape=[self.num_visible], dtype=tf.float32))
        return weights

    def _create_goodness(self):
        association = tf.matmul(tf.transpose(self.v), self.h)
        neg_energy = tf.reduce_mean(tf.mul(self.weights['w'], association))
        return neg_energy
    
    def sample_visible_from_hidden(self, hidden):
        logit = tf.matmul(hidden, tf.transpose(self.weights['w']))
        vprob = self.trans_func(logit)
        vstate = sample_prob_dist(vprob, self.vrand)
        return vprob, vstate
    
    def sample_hidden_from_visible(self, visible):
        logit = tf.matmul(visible, self.weights['w'])
        hprob = self.trans_func(logit)
        hstate = sample_prob_dist(hprob, self.hrand)
        return hprob, hstate
    
    def gibbs_sampling_step(self, visible):
        # postive phase
        pos_hprob, pos_hstate = self.sample_hidden_from_visible(visible)
        
        # negative phase
        neg_vprob, neg_vstate = self.sample_visible_from_hidden(pos_hprob)
        neg_hprob, neg_hstate = self.sample_hidden_from_visible(neg_vprob)
        return pos_hprob, pos_hstate, neg_vprob, neg_vstate, neg_hprob, neg_hstate
    
    def _create_cd1(self):
        pos_hprob, pos_hstate, neg_vprob, neg_vstate, neg_hprob, neg_hstate = self.gibbs_sampling_step(self.v)
        
        pos_association = tf.matmul(tf.transpose(self.v), pos_hstate)
        neg_association = tf.matmul(tf.transpose(neg_vstate), neg_hstate)
        
        gradient_w = self.lr * tf.sub(pos_association, neg_association) / self.batch_size
        update_w = tf.assign_add(self.weights['w'], gradient_w)
        
        # gradient_bh = self.lr * tf.reduce_mean(tf.sub(pos_hprob, neg_hprob), 0)
        # update_bh = tf.assign_add(self.weights['bh'], gradient_bh)
        
        # gradient_bv = self.lr * tf.reduce_mean(tf.sub(self.v, neg_vprob), 0)
        # update_bv = tf.assign_add(self.weights['bv'], gradient_bv)
        
        # mean squared error
        loss = 0.5 * tf.reduce_mean(tf.square(tf.sub(neg_vstate, self.v)))

        return [loss, update_w]
        # return [loss, update_w, update_bh, update_bv]
    
    def run_train_step(self, V, Vrand, Hrand):
        return self.sess.run(self.updates,
                             feed_dict={self.v: V, self.vrand: Vrand, self.hrand: Hrand})
    
    def calculate_goodness(self, V, Hrand):
        hstates = self.encodeDataset(V, Hrand)
        return self.sess.run(self.goodness,
                             feed_dict={self.v: V, self.h: hstates})
    
    def encode_dataset(self, V, Hrand):
        return self.sess.run(self.encode,
                             feed_dict={self.v: V, self.hrand: Hrand})
    
    def reconstruct_dataset(self, H, Vrand):
        return self.sess.run(self.reconstruct,
                             feed_dict={self.h: H, self.vrand: Vrand})
    
    def reconstruct_loss(self, V, Vrand, Hrand):
        return self.sess.run(self.loss,
                             feed_dict={self.v: V, self.vrand: Vrand, self.hrand: Hrand})


# ### Run Objected RBM

# In[36]:

num_samples = train_dataset.shape[0]
feature_size = train_dataset.shape[1]
num_labels = train_labels.shape[1]
num_hidden_rbm = 100
rbm_lr = 0.1

batch_size = 5000
num_epochs = 400.0
num_steps = int(num_samples * num_epochs / batch_size)
print(num_steps)

rbm = RestrictedBoltzmannMachine(feature_size, num_hidden_rbm, batch_size, rbm_lr)
print('Restricted Boltzmann Machine built')


# In[37]:

for step in range(num_steps):
    offset = (batch_size * step) % (num_samples - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_vrand = np.random.binomial(1, 0.5, size=[batch_data.shape[0], feature_size])
    batch_hrand = np.random.binomial(1, 0.5, size=(batch_data.shape[0], num_hidden_rbm))
    l, _ = rbm.run_train_step(batch_data, batch_vrand, batch_hrand)

    if step % (num_steps / 100) == 0:
        print("Minibatch reconstruction loss at step %d:\t%f" % (step, l))


print('Restricted Boltzmann Machine trained')
# train_loss = rbm.calculate_goodness(train_dataset)
# print("Trainset goodness: %f" % train_loss)
vrand = np.random.binomial(1, 0.5, size=(test_dataset.shape[0], feature_size))
hrand = np.random.binomial(1, 0.5, size=(test_dataset.shape[0], num_hidden_rbm))
test_loss = rbm.reconstruct_loss(test_dataset, vrand, hrand)
print("Testset reconstruction error: %f" % test_loss)


# In[38]:

hrand = np.random.binomial(1, 0.5, size=(train_dataset.shape[0], num_hidden_rbm))
encoded_train_dataset = rbm.encode_dataset(train_dataset, hrand)

hrand = np.random.binomial(1, 0.5, size=(valid_dataset.shape[0], num_hidden_rbm))
encoded_valid_dataset = rbm.encode_dataset(valid_dataset, hrand)

hrand = np.random.binomial(1, 0.5, size=(test_dataset.shape[0], num_hidden_rbm))
encoded_test_dataset = rbm.encode_dataset(test_dataset, hrand)
print('Dataset encoded')

print('Encoded training set', encoded_train_dataset.shape, train_labels.shape)
print('Encoded validation set', encoded_valid_dataset.shape, valid_labels.shape)
print('Encoded test set', encoded_test_dataset.shape, test_labels.shape)


# ### Multilayer Perceptron with 1 hidden layer

# In[39]:

init_learning_rate = 0.99
decay_steps = 100000

decay_base = 0.96
num_hidden_classify = 256
beta = 0.001
batch_size = 100000

graph_class = tf.Graph()
with graph_class.as_default():
    tf_train_dataset = tf.placeholder(tf.float32, [batch_size, num_hidden_rbm])
    tf_train_labels = tf.placeholder(tf.float32, [batch_size, num_labels])
    tf_valid_dataset = tf.placeholder(tf.float32, [None, num_hidden_rbm])
    tf_test_dataset = tf.placeholder(tf.float32, [None, num_hidden_rbm])
    
    tf_all_train_dataset = tf.placeholder(tf.float32, [None, num_hidden_rbm])
    keep_prob = tf.placeholder(tf.float32)
    
    W1_class = tf.Variable(tf.truncated_normal([num_hidden_rbm, num_hidden_classify], stddev=0.12))
    b1_class = tf.Variable(tf.zeros(num_hidden_classify))
    W2_class = tf.Variable(tf.truncated_normal([num_hidden_classify, num_labels], stddev=0.08))
    b2_class = tf.Variable(tf.zeros(num_labels))
    
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


# ### Train the single hidden layer

# In[40]:

with tf.Session(graph=graph_class) as classify_session:
    tf.initialize_all_variables().run()
    print('Initialized')
    num_steps = 301
    for step in range(num_steps):
        offset = (batch_size * step) % (train_labels.shape[0] - batch_size)
        batch_data = encoded_train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels,
                     keep_prob: 0.5}
        _, l, train_predictions = classify_session.run([optimizer_class, class_loss, train_predict],
                                              feed_dict=feed_dict)
        if step % 100 == 0:
            print("Minibatch loss at step %d: \t%f" % (step, l))
            print("Minibatch train accuracy: \t\t%f%%" % accuracy(train_predictions, batch_labels))
            feed_dict = {tf_valid_dataset: encoded_valid_dataset,
                         keep_prob: 1.0}
            print("Minibatch validation accuracy: \t\t\t\t%f%%" %
                  accuracy(valid_predict.eval(feed_dict=feed_dict), valid_labels))
        
    print('Softmax classifier trained')
    feed_dict_train = {tf_all_train_dataset: encoded_train_dataset,
                       keep_prob: 1.0}
    all_train_predictions = all_train_predict.eval(feed_dict=feed_dict_train)

    feed_dict_valid = {tf_valid_dataset: encoded_valid_dataset,
                      keep_prob: 1.0}
    valid_predictions = valid_predict.eval(feed_dict=feed_dict_valid)

    feed_dict_test = {tf_test_dataset: encoded_test_dataset,
                      keep_prob: 1.0}
    test_predictions = test_predict.eval(feed_dict=feed_dict_test)

    print('Classifier trained')


# ### Print accuracy and claasification table

# In[ ]:

def measure_prediction(predictions, labels, dataset_name='Test'):
    print("%sset accuracy: %f%%" % (dataset_name, accuracy(predictions, labels)))
    headers = [str(i) for i in range(labels.shape[1])]
    class_table = compute_classification_table(predictions, labels)
    print(tabulate(class_table, headers))
    correct_percentage(class_table)
    
measure_prediction(all_train_predictions, train_labels, 'Train')
measure_prediction(valid_predictions, valid_labels, 'Valid')
measure_prediction(test_predictions, test_labels, 'Test')


# In[ ]:



