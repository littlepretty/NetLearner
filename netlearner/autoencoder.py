from __future__ import print_function
import numpy as np
import tensorflow as tf
from utils import xavier_init


class Autoencoder(object):
    def __init__(self, feature_size, encode_size, encode_lr=0.01, beta=0.003,
                 transfer_func=tf.nn.softplus,
                 optimizer=tf.train.GradientDescentOptimizer):
        self.feature_size = feature_size
        self.encode_size = encode_size
        self.reg_factor = beta
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
        self.loss = tf.add(self._create_loss(), tf.mul(self.regterm, self.reg_factor))
        self.optimizer = optimizer(encode_lr).minimize(self.loss)

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

        all_weights['w2'] = tf.Variable(tf.zeros([self.encode_size,
                                                  self.feature_size],
                                                 dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.feature_size],
                                                 dtype=tf.float32))

        return all_weights

    def _create_loss(self):
        return 0.5 * tf.reduce_mean(tf.pow(
            tf.sub(self.reconstruction, self.x), 2.0))
        # self.cross_entropy_loss = tf.reduce_mean(
        # tf.nn.sigmoid_cross_entropy_with_logits(
        # self.reconstruction, self.x))

    def _create_regterm(self):
        l2regterm = tf.nn.l2_loss([0.0])
        for name, param in self.weights.items():
            l2regterm = tf.add(l2regterm, tf.nn.l2_loss(param))
        return l2regterm

    def partial_fit(self, X):
        opt, loss, reg = self.sess.run([self.optimizer,
                                        self.loss,
                                        self.regterm],
                                       feed_dict={self.x: X})
        return loss, reg

    def calc_total_loss(self, X):
        return self.sess.run(self.loss, feed_dict={self.x: X})

    def encode_dataset(self, X):
        return self.sess.run(self.encode, feed_dict={self.x: X})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction,
                             feed_dict={self.x: X})

    def get_encode_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_encode_biases(self):
        return self.sess.run(self.weights['b1'])

    def train(self, train_dataset, batch_size, num_steps):
        display_step = num_steps / 10
        print('Training for %d steps' % num_steps)

        for step in range(num_steps):
            offset = (batch_size * step) % (train_dataset.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]

            loss, reg = self.partial_fit(batch_data)
            if step % display_step == 0:
                print("Minibatch loss at step %d:\t%f(regterm=%f)" % (step, loss, reg))

        print('Autoencoder trained')
        train_loss = self.calc_total_loss(train_dataset)
        print("Trainset decode loss: %f" % train_loss)


class SparseAutoencoder(Autoencoder):
    def __init__(self, feature_size, encode_size, encode_lr=0.01,
                 beta=0.003, transfer_func=tf.nn.softplus,
                 optimizer=tf.train.GradientDescentOptimizer):
        super(self).__init__(feature_size, encode_size, encode_lr,
                             beta, transfer_func,
                             optimizer)

    def _create_regterm(self):
        l1regterm = tf.reduce_sum(tf.abs([0.0]))
        for name, param in self.weights.items():
            l1regterm = tf.reduce_sum(tf.abs(param))
        return l1regterm
