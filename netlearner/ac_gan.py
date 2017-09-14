from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils import next_batch, plot_samples, plot_V, xavier_init
from time import localtime, strftime


class AuxiliaryClassifierGAN(object):
    def __init__(self, noise_dim, input_dim, label_dim,
                 G_hidden_layer, D_hidden_layer,
                 init_lr, decay_steps,
                 optimizer=tf.train.AdamOptimizer, name='AC-GAN'):
        self.noise_dim = noise_dim
        self.input_dim = input_dim
        self.label_dim = label_dim
        self.G_hidden_layer = G_hidden_layer
        self.D_hidden_layer = D_hidden_layer

        self.theta_G, self.theta_D = self._create_variables()

        global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(init_lr, global_step,
                                             decay_steps, 0.96, staircase=True)
        self.Z = tf.placeholder(tf.float64, [None, noise_dim], name='noise')
        self.X = tf.placeholder(tf.float64, [None, input_dim], name='input')
        self.Y = tf.placeholder(tf.float64, [None, label_dim], name='label')
        self.keep_prob = tf.placeholder(tf.float64, name='keep_prob')

        self.G_sample = self.generator(self.Z, self.Y)
        D_real, C_real_logits = self.discriminator(self.X)
        D_fake, C_fake_logits = self.discriminator(self.G_sample)

        self.S_neg = tf.reduce_mean(-tf.log(D_real) - tf.log(1.0 - D_fake))
        self.C_neg = self.cross_entropy(C_real_logits, self.Y) + \
            self.cross_entropy(C_fake_logits, self.Y)

        self.D_V_neg = self.S_neg + self.C_neg
        self.G_V = tf.reduce_mean(-tf.log(D_fake)) + self.C_neg

        self.D_solver = optimizer(learning_rate=self.lr).minimize(
            self.D_V_neg, var_list=self.theta_D, global_step=global_step)
        self.G_solver = optimizer(learning_rate=self.lr).minimize(
            self.G_V, var_list=self.theta_G, global_step=global_step)

        time_str = strftime("%b-%d-%Y-%H-%M-%S", localtime())
        self.dirname = name + '/Run-' + time_str
        self.train_writer = tf.summary.FileWriter(self.dirname)
        self._create_summaries()
        self.merged_summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print('%s build and initialized' % name)

    def cross_entropy(self, predict, label):
        ce = tf.nn.softmax_cross_entropy_with_logits(logits=predict,
                                                     labels=label)
        return tf.reduce_mean(ce)

    def _create_variables(self):
        G_W1 = tf.Variable(xavier_init(self.noise_dim + self.label_dim,
                                       self.G_hidden_layer))
        G_b1 = tf.Variable(tf.zeros(shape=[self.G_hidden_layer],
                                    dtype=tf.float64))

        G_W2 = tf.Variable(xavier_init(self.G_hidden_layer,
                                       self.input_dim))
        G_b2 = tf.Variable(tf.zeros(shape=[self.input_dim],
                                    dtype=tf.float64))
        theta_G = {'G_W1': G_W1, 'G_b1': G_b1,
                   'G_W2': G_W2, 'G_b2': G_b2}

        D_W1 = tf.Variable(xavier_init(self.input_dim,
                                       self.D_hidden_layer))
        D_b1 = tf.Variable(tf.zeros(shape=[self.D_hidden_layer],
                                    dtype=tf.float64))

        D_W2_gan = tf.Variable(xavier_init(self.D_hidden_layer, 1))
        D_b2_gan = tf.Variable(tf.zeros(shape=[1], dtype=tf.float64))

        D_W2_aux = tf.Variable(xavier_init(self.D_hidden_layer,
                                           self.label_dim))
        D_b2_aux = tf.Variable(tf.zeros(shape=[self.label_dim],
                                        dtype=tf.float64))

        theta_D = {'D_W1': D_W1, 'D_b1': D_b1,
                   'D_W2_gan': D_W2_gan, 'D_b2_gan': D_b2_gan,
                   'D_W2_aux': D_W2_aux, 'D_b2_aux': D_b2_aux}

        return theta_G, theta_D

    def generator(self, z, y):
        G_W1, G_b1 = self.theta_G['G_W1'], self.theta_G['G_b1']
        G_W2, G_b2 = self.theta_G['G_W2'], self.theta_G['G_b2']

        x = tf.concat(axis=1, values=[z, y])
        h1 = tf.nn.relu(tf.matmul(x, G_W1) + G_b1)
        G_logit = tf.matmul(h1, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_logit)
        return G_prob

    def discriminator(self, x, keep_prob=1.0):
        D_W1, D_b1 = self.theta_D['D_W1'], self.theta_D['D_b1']
        D_W2_gan, D_b2_gan = self.theta_D['D_W2_gan'], self.theta_D['D_b2_gan']
        D_W2_aux, D_b2_aux = self.theta_D['D_W2_aux'], self.theta_D['D_b2_aux']

        h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        dropout = tf.nn.dropout(h1, self.keep_prob)

        D_gan_logit = tf.matmul(dropout, D_W2_gan) + D_b2_gan
        D_gan_prob = tf.nn.sigmoid(D_gan_logit)

        D_aux_logits = tf.matmul(dropout, D_W2_aux) + D_b2_aux
        # predicted label is returned as logit
        return D_gan_prob, D_aux_logits

    def sample_noise(self, size):
        return np.random.normal(loc=0.0, scale=1.0, size=size)

    def _create_summaries(self):
        tf.summary.scalar('data dimension', self.input_dim)
        tf.summary.scalar('label dimension', self.label_dim)
        tf.summary.scalar('noise prior dimension', self.noise_dim)
        tf.summary.scalar('learning rate', self.lr)
        tf.summary.scalar('Genearator hidden size', self.G_hidden_layer)
        tf.summary.scalar('Discriminator hidden size', self.D_hidden_layer)
        tf.summary.scalar('S', -self.S_neg)
        tf.summary.scalar('C', -self.C_neg)
        tf.summary.scalar('DV', -self.D_V_neg)  # should increase
        tf.summary.scalar('GV', self.G_V)  # should decrease
        # tf.summary.histogram('X_fake', self.G_sample)
        # tf.summary.histogram('X_real', self.X)
        # tf.summary.image('Geneated MNIST images',
                         # tf.reshape(self.G_sample, [-1, 28, 28, 1]),
                         # max_outputs=10)

    def make_summary(self, step, X, Z, Y, keep_prob):
        summaries = self.sess.run(self.merged_summary,
                                  feed_dict={self.X: X,
                                             self.Z: Z,
                                             self.Y: Y,
                                             self.keep_prob: keep_prob})
        self.train_writer.add_summary(summaries, step)

    def train(self, batch_size, train_dataset, train_labels,
              num_steps, keep_prob=0.8):
        display_step = num_steps // 20
        summary_step = num_steps // 100

        perm = np.random.permutation(train_dataset.shape[0])
        X = train_dataset[perm, :]
        Y = train_labels[perm, :]

        # Use fixed Z to generate samples
        display_Z = self.sample_noise([100, self.noise_dim])
        labels = np.ones(10)
        display_Y = np.tile(np.diag(labels), [10, 1])

        fig_index = 0
        print('Training GAN for %d steps' % num_steps)
        D_history = []
        G_history = []
        for step in xrange(num_steps):
            # use next different batches
            batch_X = next_batch(X, step, batch_size)
            batch_Y = next_batch(Y, step, batch_size)
            batch_Z_D = self.sample_noise([batch_size, self.noise_dim])
            _, D_V_neg = self.sess.run([self.D_solver, self.D_V_neg],
                                       feed_dict={self.X: batch_X,
                                                  self.Y: batch_Y,
                                                  self.Z: batch_Z_D,
                                                  self.keep_prob: keep_prob})
            batch_Z_G = self.sample_noise([batch_size, self.noise_dim])
            _, G_V = self.sess.run([self.G_solver, self.G_V],
                                   feed_dict={self.X: batch_X,
                                              self.Y: batch_Y,
                                              self.Z: batch_Z_G,
                                              self.keep_prob: keep_prob})

            if step % display_step == 0:
                print('Batch(%d cases) value function at step %d' %
                      (batch_X.shape[0], step))
                print('V(D) = %.6f, V(G) = %.6f' % (-D_V_neg, G_V))
                samples = self.sess.run(self.G_sample,
                                        feed_dict={self.Z: display_Z,
                                                   self.Y: display_Y,
                                                   self.keep_prob: 1.0})
                plot_samples(samples, self.dirname, fig_index, size=10)
                fig_index += 1

            if step % summary_step == 0:
                D_history.append(-D_V_neg)
                G_history.append(G_V)
                self.make_summary(step, batch_X, batch_Z_G, batch_Y, keep_prob)

        Z_D = self.sample_noise([X.shape[0], self.noise_dim])
        D_V_neg = self.sess.run(self.D_V_neg,
                                feed_dict={self.X: X,
                                           self.Y: Y,
                                           self.Z: Z_D,
                                           self.keep_prob: 1.0})
        Z_G = self.sample_noise([X.shape[0], self.noise_dim])
        G_V = self.sess.run(self.G_V, feed_dict={self.X: X,
                                                 self.Y: Y,
                                                 self.Z: Z_G,
                                                 self.keep_prob: 1.0})
        print('Finish training\nV(D) = %.6f, V(G) = %.6f' % (-D_V_neg, G_V))
        self.make_summary(num_steps, X, Z_G, Y, keep_prob)
        plot_V(self.dirname, D_history, G_history)

    def close(self):
        self.sess.close()


class ACGANTwoLayers(AuxiliaryClassifierGAN):
    def __init__(self, noise_dim, input_dim, label_dim,
                 G_hidden_layer, D_hidden_layer,
                 init_lr, decay_steps,
                 optimizer=tf.train.AdamOptimizer, name='AC-GAN-2Layers'):
        super(ACGANTwoLayers, self).__init__(noise_dim, input_dim, label_dim,
                                             G_hidden_layer, D_hidden_layer,
                                             init_lr, decay_steps,
                                             optimizer, name)

    def _create_variables(self):
        G_W1 = tf.Variable(xavier_init(self.noise_dim + self.label_dim,
                                       self.G_hidden_layer[0]))
        G_b1 = tf.Variable(tf.zeros(shape=[self.G_hidden_layer[0]],
                                    dtype=tf.float64))
        G_W2 = tf.Variable(xavier_init(self.G_hidden_layer[0],
                                       self.G_hidden_layer[1]))
        G_b2 = tf.Variable(tf.zeros(shape=[self.G_hidden_layer[1]],
                                    dtype=tf.float64))

        G_W3 = tf.Variable(xavier_init(self.G_hidden_layer[1],
                                       self.input_dim))
        G_b3 = tf.Variable(tf.zeros(shape=[self.input_dim],
                                    dtype=tf.float64))
        theta_G = {'G_W1': G_W1, 'G_b1': G_b1,
                   'G_W2': G_W2, 'G_b2': G_b2,
                   'G_W3': G_W3, 'G_b3': G_b3, }

        D_W1 = tf.Variable(xavier_init(self.input_dim,
                                       self.D_hidden_layer[0]))
        D_b1 = tf.Variable(tf.zeros(shape=(self.D_hidden_layer[0]),
                                    dtype=tf.float64))
        D_W2 = tf.Variable(xavier_init(self.D_hidden_layer[0],
                                       self.D_hidden_layer[1]))
        D_b2 = tf.Variable(tf.zeros(shape=(self.D_hidden_layer[1]),
                                    dtype=tf.float64))

        D_W3_gan = tf.Variable(xavier_init(self.D_hidden_layer[1], 1))
        D_b3_gan = tf.Variable(tf.zeros(shape=[1], dtype=tf.float64))

        D_W3_aux = tf.Variable(xavier_init(self.D_hidden_layer[1],
                                           self.label_dim))
        D_b3_aux = tf.Variable(tf.zeros(shape=[self.label_dim],
                                        dtype=tf.float64))

        theta_D = {'D_W1': D_W1, 'D_b1': D_b1,
                   'D_W2': D_W2, 'D_b2': D_b2,
                   'D_W3_gan': D_W3_gan, 'D_b3_gan': D_b3_gan,
                   'D_W3_aux': D_W3_aux, 'D_b3_aux': D_b3_aux}

        return theta_G, theta_D

    def generator(self, z, y):
        G_W1, G_b1 = self.theta_G['G_W1'], self.theta_G['G_b1']
        G_W2, G_b2 = self.theta_G['G_W2'], self.theta_G['G_b2']
        G_W3, G_b3 = self.theta_G['G_W3'], self.theta_G['G_b3']

        x = tf.concat(axis=1, values=[z, y])
        h1 = tf.nn.relu(tf.matmul(x, G_W1) + G_b1)
        h2 = tf.nn.relu(tf.matmul(h1, G_W2) + G_b2)
        G_logit = tf.matmul(h2, G_W3) + G_b3
        G_prob = tf.nn.sigmoid(G_logit)
        return G_prob

    def discriminator(self, x, keep_prob=0.8):
        D_W1, D_b1 = self.theta_D['D_W1'], self.theta_D['D_b1']
        D_W2, D_b2 = self.theta_D['D_W2'], self.theta_D['D_b2']
        D_W3_gan, D_b3_gan = self.theta_D['D_W3_gan'], self.theta_D['D_b3_gan']
        D_W3_aux, D_b3_aux = self.theta_D['D_W3_aux'], self.theta_D['D_b3_aux']

        h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
        dropout1 = tf.nn.dropout(h1, self.keep_prob)
        h2 = tf.nn.relu(tf.matmul(dropout1, D_W2) + D_b2)
        dropout2 = tf.nn.dropout(h2, self.keep_prob)

        D_gan_logit = tf.matmul(dropout2, D_W3_gan) + D_b3_gan
        D_gan_prob = tf.nn.sigmoid(D_gan_logit)
        # predicted label is returned as logit
        D_aux_logit = tf.matmul(dropout2, D_W3_aux) + D_b3_aux

        return D_gan_prob, D_aux_logit

    def _create_summaries(self):
        tf.summary.scalar('data dimension', self.input_dim)
        tf.summary.scalar('label dimension', self.label_dim)
        tf.summary.scalar('noise prior dimension', self.noise_dim)
        tf.summary.scalar('learning rate', self.lr)
        tf.summary.scalar('Genearator h1 size', self.G_hidden_layer[0])
        tf.summary.scalar('Genearator h2 size', self.G_hidden_layer[1])
        tf.summary.scalar('Discriminator h1 size', self.D_hidden_layer[0])
        tf.summary.scalar('Discriminator h2 size', self.D_hidden_layer[0])
        tf.summary.scalar('S', -self.S_neg)
        tf.summary.scalar('C', -self.C_neg)
        tf.summary.scalar('DV', -self.D_V_neg)  # should increase
        tf.summary.scalar('GV', self.G_V)  # should decrease
