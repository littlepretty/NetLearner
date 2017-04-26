from __future__ import print_function
import numpy as np
import tensorflow as tf
from utils import xavier_init, accuracy, measure_prediction, get_batch
from netlearner.rbm import RestrictedBoltzmannMachine


class StackedRBM(object):
    def __init__(self, feature_size, rbm_layer_sizes, num_labels,
                 rbm_lr=0.1, rbm_trans_func=tf.nn.sigmoid, rbm_batch_size=400,
                 ft_trans_func=tf.nn.relu, ft_reg_func=tf.nn.l2_loss,
                 ft_lr=0.0001, ft_reg_factor=0.0001,
                 ft_optimizer=tf.train.GradientDescentOptimizer):
        self.feature_size = feature_size
        self.num_labels = num_labels

        self.rbm_layer_sizes = rbm_layer_sizes
        self.rbm_trans_func = rbm_trans_func
        self.rbm_lr = rbm_lr
        self.rbm_batch_size = rbm_batch_size
        self.rbms = []
        input_size = self.feature_size
        for i, hidden_size in enumerate(self.rbm_layer_sizes):
            rbm = RestrictedBoltzmannMachine(input_size, hidden_size, rbm_batch_size,
                                             self.rbm_lr, self.rbm_trans_func, name='rbm%d' % i)
            input_size = hidden_size
            self.rbms.append(rbm)

        self.x = tf.placeholder(tf.float32, [None, feature_size])
        self.ft_target = tf.placeholder(tf.float32, [None, feature_size])
        self.ft_trans_func = ft_trans_func
        self.ft_keep_prob = tf.placeholder(tf.float32)
        self.ft_reg_func = ft_reg_func
        self.ft_reg_factor = ft_reg_factor
        self.ft_lr = ft_lr
        self.ft_params = None
        self.ft_loss = None
        self.ft_optimizer = ft_optimizer

        self.sess = None

    def run_pretrain(self, train_dataset, train_labels, batch_sizes, num_steps):
        input_data = train_dataset
        for i, rbm in enumerate(self.rbms):
            # rbm.train(input_data, num_steps[i])
            rbm.train_with_labels(input_data, train_labels, batch_sizes[i], num_steps[i])
            hrand = np.random.random((input_data.shape[0], rbm.num_hidden))
            input_data = rbm.encode_dataset(input_data, hrand)

        print('All individual RBMs trained')

    def _create_ft_weights(self):
        weights = dict()
        i = 0
        for hsize in self.rbm_layer_sizes:
            weights['w%d' % i] = tf.Variable(self.rbms[i].get_weights('w'))
            weights['b%d' % i] = tf.Variable(tf.zeros(shape=[hsize], dtype=tf.float32))
            i += 1
        # symmetric upper layers
        j = len(self.rbm_layer_sizes) - 1
        unrolled_layer_sizes = self.rbm_layer_sizes[::-1]
        unrolled_layer_sizes = unrolled_layer_sizes[1:] + [self.feature_size]
        print(unrolled_layer_sizes)
        for hsize in unrolled_layer_sizes:
            weights['w%d' % i] = tf.Variable(tf.transpose(self.rbms[j].get_weights('w')))
            weights['b%d' % i] = tf.Variable(tf.zeros(shape=[hsize], dtype=tf.float32))
            i += 1
            j -= 1

        return weights

    def _create_unrolled_network(self):
        next_input = self.x
        for i in range(len(self.rbm_layer_sizes) * 2):
            logits = tf.add(tf.matmul(next_input, self.ft_params['w%d' % i]),
                            self.ft_params['b%d' % i])
            activity = self.ft_trans_func(logits)
            next_input = activity
            # next_input = tf.nn.dropout(activity, self.ft_keep_prob)
            if i == len(self.rbm_layer_sizes) - 1:
                encode = next_input

        return next_input, encode

    def _create_regterm(self):
        regterm = self.ft_reg_func([0.0])
        for i in range(len(self.rbm_layer_sizes) * 2):
            regterm = tf.add(regterm, self.ft_reg_func(self.ft_params['w%d' % i]))

        return tf.mul(self.ft_reg_factor, regterm)

    def unrolling(self):
        self.ft_params = self._create_ft_weights()

        self.reconstruct, self.encode = self._create_unrolled_network()
        #self.finetune_loss = tf.reduce_mean(
        #    tf.nn.softmax_cross_entropy_with_logits(self.reconstruct, self.ft_target))
        self.finetune_loss = 0.5 * tf.reduce_mean(tf.pow(tf.sub(self.reconstruct, self.ft_target), 2.0))
        self.ft_regterm = self._create_regterm()
        self.ft_loss = self.finetune_loss + self.ft_regterm
        self.optimizer = self.ft_optimizer(self.ft_lr).minimize(self.ft_loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print('Unrolled all RBMs and initialized')

    def unsupervise_fit(self, X, prob=1.0):
        opt, loss, reg = self.sess.run(
            [self.optimizer, self.ft_loss, self.ft_regterm],
            feed_dict={self.x: X, self.ft_target: X, self.ft_keep_prob: prob})
        return loss, reg

    def encode_dataset(self, X, prob=1.0):
        return self.sess.run(self.encode,
                             feed_dict={self.x: X,
                                        self.ft_keep_prob: prob})

    def calc_total_loss(self, X, prob=1.0):
        return self.sess.run(self.ft_loss, feed_dict={self.x: X,
                                                      self.ft_target: X,
                                                      self.ft_keep_prob: prob})

    def run_fine_tuning(self, train_dataset, train_labels, batch_size, num_steps, keep_prob):
        display_step = num_steps / 10
        print('Training for %d steps' % num_steps)

        train_perm = np.random.permutation(train_dataset.shape[0])
        train_dataset = train_dataset[train_perm, :]
        train_labels = train_labels[train_perm, :]

        y = np.argmax(train_labels, 1)
        X = np.array([train_dataset[y == i, :] for i in range(self.num_labels)])
        Y = np.array([train_labels[y == i, :] for i in range(self.num_labels)])

        if self.num_labels > 2:
            batch_size /= (self.num_labels - 1)
            for step in range(num_steps):
                train0, label0 = get_batch(X[0], Y[0], step, batch_size)
                train1, label1 = get_batch(X[1], Y[1], step, batch_size)
                train2, label2 = get_batch(X[2], Y[2], step, batch_size)
                # train3 = X[3][np.random.choice(X[3].shape[0], 50), :]
                train3 = X[3][:, :]
                # label3 = Y[3][:, :]
                train4, label4 = get_batch(X[4], Y[4], step, batch_size)

                batch_data = np.concatenate((train0, train1, train2, train3, train4), axis=0)
                perm = np.random.permutation(batch_data.shape[0])
                batch_data = batch_data[perm, :]

                loss, reg = self.unsupervise_fit(batch_data, keep_prob)
                if step % display_step == 0:
                    print("Minibatch(%d cases) loss at step %d:\t%f(regterm=%f)"
                          % (batch_data.shape[0], step, loss, reg))
        else:
            for step in range(num_steps):
                train0, label0 = get_batch(X[0], Y[0], step, batch_size)
                train1, label1 = get_batch(X[1], Y[1], step, batch_size)

                batch_data = np.concatenate((train0, train1), axis=0)
                batch_labels = np.concatenate((label0, label1), axis=0)
                # perm = np.random.permutation(batch_data.shape[0])
                # batch_data = batch_data[perm, :]
                # batch_labels = batch_labels[perm, :]

                loss, reg = self.unsupervise_fit(batch_data, keep_prob)
                if step % display_step == 0:
                    print("Minibatch(%d cases) total loss at step %d:\t%f(regterm=%f)"
                          % (batch_data.shape[0], step, loss, reg))

        print('Fine-tuning phase finished')
        train_loss = self.calc_total_loss(train_dataset)
        print("Trainset total loss: %f" % train_loss)

    def train(self, train_dataset, train_labels, rbm_batch_sizes, rbm_num_steps,
              ft_batch_size, ft_num_steps, ft_keep_prob=0.72):
        self.run_pretrain(train_dataset, train_labels, rbm_batch_sizes, rbm_num_steps)
        self.unrolling()
        self.run_fine_tuning(train_dataset, train_labels, ft_batch_size, ft_num_steps, ft_keep_prob)