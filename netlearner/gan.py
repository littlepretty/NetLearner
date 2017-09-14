from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils import next_batch, plot_samples, plot_V, xavier_init
from time import localtime, strftime


def another_loss():
    D_V_real_neg = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_real_logit,
        labels=tf.ones_like(D_real_logit))
    D_V_fake_neg = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_logit,
        labels=tf.zeros_like(D_fake_logit))
    G_V = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=D_fake_logit,
        labels=tf.ones_like(D_fake_logit))
    self.G_V = tf.reduce_mean(G_V)


class GenerativeAdversarialNets(object):
    def __init__(self, noise_dim, input_dim,
                 G_hidden_layer, D_hidden_layer,
                 init_lr=0.001, decay_steps=10000,
                 optimizer=tf.train.AdamOptimizer, name="VanillaGAN"):
        self.noise_dim = noise_dim
        self.input_dim = input_dim
        self.G_hidden_layer = G_hidden_layer
        self.D_hidden_layer = D_hidden_layer

        self.theta_G, self.theta_D = self._create_variables()

        global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(init_lr, global_step,
                                             decay_steps, 0.96, staircase=True)
        self.Z = tf.placeholder(tf.float64, [None, noise_dim], name='noise')
        self.X = tf.placeholder(tf.float64, [None, input_dim], name='input')
        self.keep_prob = tf.placeholder(tf.float64, name='keep_prob')

        self.G_sample = self.generator(self.Z)
        D_real_prob, D_real_logit = self.discriminator(self.X)
        D_fake_prob, D_fake_logit = self.discriminator(self.G_sample)
        D_V_real_neg = -tf.log(D_real_prob)
        D_V_fake_neg = -tf.log(1.0 - D_fake_prob)

        self.D_V_real_neg = tf.reduce_mean(D_V_real_neg)
        self.D_V_fake_neg = tf.reduce_mean(D_V_fake_neg)
        self.D_V_neg = self.D_V_real_neg + self.D_V_fake_neg
        self.G_V = tf.reduce_mean(-tf.log(D_fake_prob))
        """
        Maximize D_V while minimize G_V
        """
        self.D_solver = optimizer(learning_rate=self.lr).minimize(
            self.D_V_neg, var_list=self.theta_D)
        self.G_solver = optimizer(learning_rate=self.lr).minimize(
            self.G_V, var_list=self.theta_G)

        time_str = strftime("%b-%d-%Y-%H-%M-%S", localtime())
        self.dirname = name + '/Run-' + time_str
        self.train_writer = tf.summary.FileWriter(self.dirname)
        self._create_summaries(D_real_prob, D_fake_prob)
        self.merged_summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print('%s build and initialized' % name)

    def tn_init(self, size):
        return tf.truncated_normal(shape=size, stddev=0.1)

    def _create_variables(self):
        G_W1 = tf.Variable(xavier_init(self.noise_dim, self.G_hidden_layer))
        G_b1 = tf.Variable(tf.zeros(shape=[self.G_hidden_layer],
                                    dtype=tf.float64))

        G_W2 = tf.Variable(xavier_init(self.G_hidden_layer, self.input_dim))
        G_b2 = tf.Variable(tf.zeros(shape=[self.input_dim], dtype=tf.float64))
        theta_G = {'G_W1': G_W1, 'G_b1': G_b1,
                   'G_W2': G_W2, 'G_b2': G_b2}

        D_W1 = tf.Variable(xavier_init(self.input_dim, self.D_hidden_layer))
        D_b1 = tf.Variable(tf.zeros(shape=[self.D_hidden_layer],
                                    dtype=tf.float64))

        D_W2 = tf.Variable(xavier_init(self.D_hidden_layer, 1))
        D_b2 = tf.Variable(tf.zeros(shape=[1], dtype=tf.float64))
        theta_D = {'D_W1': D_W1, 'D_b1': D_b1,
                   'D_W2': D_W2, 'D_b2': D_b2}

        return theta_G, theta_D

    def _create_summaries(self, D_real, D_fake):
        tf.summary.scalar('data dimension', self.input_dim)
        tf.summary.scalar('noise prior dimension', self.noise_dim)
        tf.summary.scalar('Genearator hidden size', self.G_hidden_layer)
        tf.summary.scalar('Discriminator hidden size', self.D_hidden_layer)
        tf.summary.scalar('DV_real', -self.D_V_real_neg)
        tf.summary.scalar('DV_fake', -self.D_V_fake_neg)
        tf.summary.scalar('DV', -self.D_V_neg)  # should increase
        tf.summary.scalar('GV', self.G_V)  # should decrease
        # tf.summary.histogram('D_real', D_real)
        # tf.summary.histogram('D_fake', D_fake)
        # tf.summary.histogram('X_fake', self.G_sample)
        # tf.summary.histogram('X_real', self.X)
        # tf.summary.image('Geneated MNIST images',
                         # tf.reshape(self.G_sample, [-1, 28, 28, 1]),
                         # max_outputs=10)

    def make_summary(self, step, X, Z, keep_prob):
        summaries = self.sess.run(self.merged_summary,
                                  feed_dict={self.X: X,
                                             self.Z: Z,
                                             self.keep_prob: keep_prob})
        self.train_writer.add_summary(summaries, step)

    def generator(self, z):
        G_W1, G_b1 = self.theta_G['G_W1'], self.theta_G['G_b1']
        G_W2, G_b2 = self.theta_G['G_W2'], self.theta_G['G_b2']

        h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        G_logit = tf.matmul(h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_logit)
        return G_prob

    def discriminator(self, x, keep_prob=0.7):
        D_W1, D_b1 = self.theta_D['D_W1'], self.theta_D['D_b1']
        D_W2, D_b2 = self.theta_D['D_W2'], self.theta_D['D_b2']

        h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        dropout1 = tf.nn.dropout(h1, self.keep_prob)
        D_logit = tf.matmul(dropout1, D_W2) + D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit

    def sample_noise(self, size):
        return np.random.normal(loc=0.0, scale=1.0, size=size)
        # return np.random.uniform(-1.0, 1.0, size=size)

    def train(self, batch_size, train_dataset, num_steps, keep_prob=1.0):
        display_step = num_steps // 40
        summary_step = num_steps // 100
        perm = np.random.permutation(train_dataset.shape[0])
        X = train_dataset[perm, :]
        # Use fixed Z to generate samples
        display_Z = self.sample_noise([64, self.noise_dim])

        fig_index = 0
        inner_step = 0
        print('Training GAN for %d steps' % num_steps)
        D_history = []
        G_history = []
        for step in xrange(num_steps):
            for k in range(1):
                # use next different batches
                batch_X = next_batch(X, inner_step, batch_size)
                inner_step += 1
                batch_Z_D = self.sample_noise([batch_size, self.noise_dim])
                _, D_V_neg = self.sess.run([self.D_solver, self.D_V_neg],
                                           feed_dict={self.X: batch_X,
                                                      self.Z: batch_Z_D,
                                                      self.keep_prob: keep_prob})
            #  finish k steps for training D
            batch_Z_G = self.sample_noise([batch_size, self.noise_dim])
            _, G_V = self.sess.run([self.G_solver, self.G_V],
                                   feed_dict={self.Z: batch_Z_G,
                                              self.keep_prob: keep_prob})

            if step % display_step == 0:
                print('Batch(%d cases) value function at step %d' %
                      (batch_X.shape[0], step))
                print('V(D) = %.6f, V(G) = %.6f' % (-D_V_neg, G_V))
                samples = self.sess.run(self.G_sample,
                                        feed_dict={self.Z: display_Z,
                                                   self.keep_prob: 1.0})
                plot_samples(samples, self.dirname, fig_index)
                fig_index += 1

            if step % summary_step == 0:
                D_history.append(-D_V_neg)
                G_history.append(G_V)
                self.make_summary(step, batch_X, batch_Z_G, keep_prob)

        Z_D = self.sample_noise([X.shape[0], self.noise_dim])
        D_V_neg = self.sess.run(self.D_V_neg,
                                feed_dict={self.X: X,
                                           self.Z: Z_D,
                                           self.keep_prob: 1.0})

        Z_G = self.sample_noise([X.shape[0], self.noise_dim])
        G_V = self.sess.run(self.G_V,
                            feed_dict={self.Z: Z_G,
                                       self.keep_prob: 1.0})
        print('Finish training\nV(D) = %.6f, V(G) = %.6f' % (-D_V_neg, G_V))
        self.make_summary(num_steps, X, Z_G, keep_prob=1.0)
        plot_V(self.dirname, D_history, G_history)

    def close(self):
        self.sess.close()


class GANTwoLayers(GenerativeAdversarialNets):
    def __init__(self, noise_dim, input_dim,
                 G_hidden_layer, D_hidden_layer,
                 init_lr=0.001, decay_steps=10000,
                 optimizer=tf.train.AdamOptimizer, name="Vanilla2LayerGAN"):
        super(GANTwoLayers, self).__init__(noise_dim, input_dim,
                                           G_hidden_layer, D_hidden_layer,
                                           init_lr, decay_steps,
                                           optimizer, name)

    def _create_variables(self):
        G_W1 = tf.Variable(xavier_init(self.noise_dim, self.G_hidden_layer[0]))
        G_b1 = tf.Variable(tf.zeros(shape=[self.G_hidden_layer[0]],
                                    dtype=tf.float64))

        G_W2 = tf.Variable(xavier_init(self.G_hidden_layer[0],
                                       self.G_hidden_layer[1]))
        G_b2 = tf.Variable(tf.zeros(shape=[self.G_hidden_layer[1]],
                                    dtype=tf.float64))

        G_W3 = tf.Variable(xavier_init(self.G_hidden_layer[1], self.input_dim))
        G_b3 = tf.Variable(tf.zeros(shape=[self.input_dim], dtype=tf.float64))
        theta_G = {'G_W1': G_W1, 'G_b1': G_b1,
                   'G_W2': G_W2, 'G_b2': G_b2,
                   'G_W3': G_W3, 'G_b3': G_b3}

        D_W1 = tf.Variable(xavier_init(self.input_dim,
                                       self.D_hidden_layer[0]))
        D_b1 = tf.Variable(tf.zeros(shape=[self.D_hidden_layer[0]],
                                    dtype=tf.float64))

        D_W2 = tf.Variable(xavier_init(self.D_hidden_layer[0],
                                       self.D_hidden_layer[1]))
        D_b2 = tf.Variable(tf.zeros(shape=[self.D_hidden_layer[1]],
                                    dtype=tf.float64))

        D_W3 = tf.Variable(xavier_init(self.D_hidden_layer[1], 1))
        D_b3 = tf.Variable(tf.zeros(shape=[1], dtype=tf.float64))
        theta_D = {'D_W1': D_W1, 'D_b1': D_b1,
                   'D_W2': D_W2, 'D_b2': D_b2,
                   'D_W3': D_W3, 'D_b3': D_b3}

        return theta_G, theta_D

    def _create_summaries(self, D_real, D_fake):
        tf.summary.scalar('data dimension', self.input_dim)
        tf.summary.scalar('noise prior dimension', self.noise_dim)
        tf.summary.scalar('Genearator hidden size 0', self.G_hidden_layer[0])
        tf.summary.scalar('Discriminator hidden size 0', self.D_hidden_layer[0])
        tf.summary.scalar('Genearator hidden size 1', self.G_hidden_layer[1])
        tf.summary.scalar('Discriminator hidden size 1', self.D_hidden_layer[1])
        tf.summary.scalar('DV_real', -self.D_V_real_neg)
        tf.summary.scalar('DV_fake', -self.D_V_fake_neg)
        tf.summary.scalar('DV', -self.D_V_neg)  # should increase
        tf.summary.scalar('GV', self.G_V)  # should decrease

    def generator(self, z):
        G_W1, G_b1 = self.theta_G['G_W1'], self.theta_G['G_b1']
        G_W2, G_b2 = self.theta_G['G_W2'], self.theta_G['G_b2']
        G_W3, G_b3 = self.theta_G['G_W3'], self.theta_G['G_b3']

        h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
        h2 = tf.nn.relu(tf.matmul(h1, G_W2) + G_b2)
        G_logit = tf.matmul(h2, G_W3) + G_b3
        G_prob = tf.nn.sigmoid(G_logit)
        return G_prob

    def discriminator(self, x, keep_prob=0.8):
        D_W1, D_b1 = self.theta_D['D_W1'], self.theta_D['D_b1']
        D_W2, D_b2 = self.theta_D['D_W2'], self.theta_D['D_b2']
        D_W3, D_b3 = self.theta_D['D_W3'], self.theta_D['D_b3']

        h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        dropout1 = tf.nn.dropout(h1, keep_prob)
        h2 = tf.nn.relu(tf.matmul(dropout1, D_W2) + D_b2)
        dropout2 = tf.nn.dropout(h2, keep_prob)
        D_logit = tf.matmul(dropout2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit
