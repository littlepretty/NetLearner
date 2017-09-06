from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils import next_batch
# from time import localtime, strftime


class GenerativeAdversarialNets(object):
    def __init__(self, noise_dim, input_dim,
                 G_hidden_layer, D_hidden_layer,
                 trans_func=tf.nn.relu,
                 optimizer=tf.train.AdamOptimizer, name="VanillaGAN"):
        self.noise_dim = noise_dim
        self.input_dim = input_dim
        self.G_hidden_layer = G_hidden_layer
        self.D_hidden_layer = D_hidden_layer
        self.trans_func = trans_func

        self.theta_G, self.theta_D = self._create_variables()

        self.learning_rate = tf.placeholder(tf.float32, name='lr')
        self.Z = tf.placeholder(tf.float32, [None, noise_dim], name='noise')
        self.X = tf.placeholder(tf.float32, [None, input_dim], name='input')

        G_sample = self.generator(self.Z)
        D_real_prob, D_real_logit = self.discriminator(self.X)
        D_fake_prob, D_fake_logit = self.discriminator(G_sample)
        # D_loss_real = -tf.log(D_real_prob)
        # D_loss_fake = -tf.log(1 - D_fake_prob)
        D_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_real_logit,
            labels=tf.ones_like(D_real_logit))
        D_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_fake_logit,
            labels=tf.zeros_like(D_fake_logit))
        self.D_loss = tf.reduce_mean(D_loss_real) + tf.reduce_mean(D_loss_fake)

        G_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_fake_logit,
            labels=tf.ones_like(D_fake_logit))
        self.G_loss = tf.reduce_mean(G_loss)
        # self.G_loss = tf.reduce_mean(tf.log(1 - D_fake_prob))

        self.D_solver = optimizer(learning_rate=self.learning_rate).minimize(
            self.D_loss, var_list=self.theta_D)
        self.G_solver = optimizer(learning_rate=self.learning_rate).minimize(
            self.G_loss, var_list=self.theta_G)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print('ValillaGAN build and initialized')

    def xavier_init(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def _create_variables(self):
        G_W1 = tf.Variable(self.xavier_init([self.noise_dim,
                                             self.G_hidden_layer]))
        G_b1 = tf.Variable(tf.zeros(shape=[self.G_hidden_layer]))
        G_W2 = tf.Variable(self.xavier_init([self.G_hidden_layer,
                                             self.input_dim]))
        G_b2 = tf.Variable(tf.zeros(shape=[self.input_dim]))
        theta_G = {'G_W1': G_W1, 'G_b1': G_b1, 'G_W2': G_W2, 'G_b2': G_b2}

        D_W1 = tf.Variable(self.xavier_init([self.input_dim,
                                             self.D_hidden_layer]))
        D_b1 = tf.Variable(tf.zeros(shape=[self.D_hidden_layer]))
        D_W2 = tf.Variable(self.xavier_init([self.D_hidden_layer, 1]))
        D_b2 = tf.Variable(tf.zeros(shape=[1]))
        theta_D = {'D_W1': D_W1, 'D_b1': D_b1, 'D_W2': D_W2, 'D_b2': D_b2}

        return theta_G, theta_D

    def generator(self, z):
        G_W1, G_b1 = self.theta_G['G_W1'], self.theta_G['G_b1']
        G_W2, G_b2 = self.theta_G['G_W2'], self.theta_G['G_b2']

        h1 = self.trans_func(tf.matmul(z, G_W1) + G_b1)
        G_logit = tf.matmul(h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_logit)
        return G_prob

    def discriminator(self, x):
        D_W1, D_b1 = self.theta_D['D_W1'], self.theta_D['D_b1']
        D_W2, D_b2 = self.theta_D['D_W2'], self.theta_D['D_b2']

        h1 = self.trans_func(tf.matmul(x, D_W1) + D_b1)
        D_logit = tf.matmul(h1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    def sample_noise(self, size):
        return np.random.uniform(-1.0, 1.0, size=size)

    def train(self, batch_size, train_dataset, num_steps, init_lr):
        display_step = num_steps // 10
        perm = np.random.permutation(train_dataset.shape[0])
        X = train_dataset[perm, :]

        for step in xrange(num_steps):
            batch_X = next_batch(X, step, batch_size)
            batch_Z_D = self.sample_noise([batch_size, self.noise_dim])
            _, D_loss = self.sess.run([self.D_solver, self.D_loss],
                                      feed_dict={self.X: batch_X,
                                                 self.Z: batch_Z_D,
                                                 self.learning_rate: init_lr})

            batch_Z_G = self.sample_noise([batch_size, self.noise_dim])
            _, G_loss = self.sess.run([self.G_solver, self.G_loss],
                                      feed_dict={self.Z: batch_Z_G,
                                                 self.learning_rate: init_lr})

            if step % display_step == 0:
                print('Batch(%d cases) loss at step %d' % (batch_X.shape[0],
                                                           step))
                print('D loss: %.6f, G loss: %.6f' % (D_loss, G_loss))

        print('Finish Training for %d steps' % num_steps)
        Z_D = self.sample_noise([X.shape[0], self.noise_dim])
        D_loss = self.sess.run(self.D_loss,
                               feed_dict={self.X: X,
                                          self.Z: Z_D})

        Z_G = self.sample_noise([X.shape[0], self.noise_dim])
        G_loss = self.sess.run(self.G_loss,
                               feed_dict={self.Z: Z_G})
        print('D loss: %.6f, G loss: %.6f' % (D_loss, G_loss))
