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

        self.ft_trans_func = ft_trans_func
        self.ft_reg_func = ft_reg_func
        self.ft_reg_factor = ft_reg_factor
        self.ft_lr = ft_lr
        self.ft_optimizer = ft_optimizer

        self.rbms = []
        input_size = self.feature_size
        for i, hidden_size in enumerate(self.rbm_layer_sizes):
            rbm = RestrictedBoltzmannMachine(input_size, hidden_size, rbm_batch_size,
                                             self.rbm_lr, self.rbm_trans_func, name='rbm%d' % i)
            input_size = hidden_size
            self.rbms.append(rbm)

        self.x = tf.placeholder(tf.float32, [None, feature_size])
        self.t = tf.placeholder(tf.float32, [None, num_labels])
        self.ft_params = None
        self.ft_keep_prob = None
        self.ft_final_logits = None
        self.classify_loss = None
        self.ft_regterm = None
        self.ft_loss = None
        self.optimizer = None
        self.predict = None
        self.sess = None

    def run_pretrain(self, train_dataset, train_labels, batch_sizes, num_steps):
        input_data = train_dataset
        for i, rbm in enumerate(self.rbms):
            rbm.train_with_labels(input_data, train_labels, batch_sizes[i], num_steps[i])
            hrand = np.random.random((input_data.shape[0], rbm.num_hidden))
            input_data = rbm.encode_dataset(input_data, hrand)

        print('All individual RBMs trained')

    def _create_ft_weights(self):
        weights = dict()
        for (i, hsize) in enumerate(self.rbm_layer_sizes):
            if i < len(self.rbm_layer_sizes):
                weights['w%d' % i] = tf.Variable(self.rbms[i].get_weights('w'))
                weights['b%d' % i] = tf.Variable(tf.zeros(shape=[hsize], dtype=tf.float32))

        fsize = self.rbm_layer_sizes[-1]
        n = len(self.rbm_layer_sizes)
        weights['w%i' % n] = tf.Variable(xavier_init(fsize, self.num_labels))
        weights['b%d' % n] = tf.Variable(tf.zeros(shape=[self.num_labels], dtype=tf.float32))

        return weights

    def _create_forward(self):
        next_input = self.x
        all_layer_sizes = self.rbm_layer_sizes + [self.num_labels]
        for (i, hsize) in enumerate(all_layer_sizes):
            logits = tf.add(tf.matmul(next_input, self.ft_params['w%d' % i]),
                            self.ft_params['b%d' % i])
            if i < len(all_layer_sizes) - 1:
                activity = self.ft_trans_func(logits)
                next_input = tf.nn.dropout(activity, self.ft_keep_prob)
            else:
                next_input = logits
                print('Last layer is just linear logistic')

        return next_input

    def _create_regterm(self):
        regterm = self.ft_reg_func([0.0])
        all_layer_sizes = self.rbm_layer_sizes + [self.num_labels]
        for (i, size) in enumerate(all_layer_sizes):
            regterm = tf.add(regterm, self.ft_reg_func(self.ft_params['w%d' % i]))
        return tf.mul(self.ft_reg_factor, regterm)

    def unrolling(self):
        self.ft_params = self._create_ft_weights()
        self.ft_keep_prob = tf.placeholder(tf.float32)

        self.ft_final_logits = self._create_forward()

        self.classify_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.ft_final_logits, self.t))
        self.ft_regterm = self._create_regterm()
        self.ft_loss = self.classify_loss + self.ft_regterm

        self.optimizer = self.ft_optimizer(self.ft_lr).minimize(self.ft_loss)
        self.predict = tf.nn.softmax(self.ft_final_logits)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print('Stacked RBM build and initialized')

    def fit(self, X, T, prob=1.0):
        opt, loss, reg = self.sess.run(
            [self.optimizer, self.ft_loss, self.ft_regterm],
            feed_dict={self.x: X, self.t: T, self.ft_keep_prob: prob})
        return loss, reg

    def make_prediction(self, X, prob=1.0):
        return self.sess.run(self.predict,
                             feed_dict={self.x: X,
                                        self.ft_keep_prob: prob})

    def calc_classify_loss(self, X, T, prob=1.0):
        closs = self.sess.run(self.classify_loss,
                              feed_dict={self.x: X, self.t: T, self.ft_keep_prob: prob})
        return closs

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
                label3 = Y[3][:, :]
                train4, label4 = get_batch(X[4], Y[4], step, batch_size)

                batch_data = np.concatenate((train0, train1, train2, train3, train4), axis=0)
                batch_labels = np.concatenate((label0, label1, label2, label3, label4), axis=0)
                perm = np.random.permutation(batch_data.shape[0])
                batch_data = batch_data[perm, :]
                batch_labels = batch_labels[perm, :]

                loss, reg = self.fit(batch_data, batch_labels, keep_prob)
                if step % display_step == 0:
                    print("Minibatch(%d cases) loss at step %d:\t%f(regterm=%f)"
                          % (batch_labels.shape[0], step, loss, reg))
                    batch_predict = self.make_prediction(batch_data)
                    print("Minibatch train accuracy: %f%%" %
                          accuracy(batch_predict, batch_labels))
        else:
            for step in range(num_steps):
                train0, label0 = get_batch(X[0], Y[0], step, batch_size)
                train1, label1 = get_batch(X[1], Y[1], step, batch_size)

                batch_data = np.concatenate((train0, train1), axis=0)
                batch_labels = np.concatenate((label0, label1), axis=0)
                # perm = np.random.permutation(batch_data.shape[0])
                # batch_data = batch_data[perm, :]
                # batch_labels = batch_labels[perm, :]

                loss, reg = self.fit(batch_data, batch_labels, keep_prob)
                if step % display_step == 0:
                    print("Minibatch(%d cases) total loss at step %d:\t%f(regterm=%f)"
                          % (batch_labels.shape[0], step, loss, reg))
                    batch_loss = self.calc_classify_loss(batch_data, batch_labels)
                    print("Minibatch classify loss:\t%f" % batch_loss)
                    batch_predict = self.make_prediction(batch_data)
                    print("Minibatch train accuracy: %f%%" % accuracy(batch_predict, batch_labels))

        print('Fine-tuning phase finished')
        train_loss = self.calc_classify_loss(train_dataset, train_labels)
        train_predict = self.make_prediction(train_dataset)
        print("Trainset total loss: %f" % train_loss)
        measure_prediction(train_predict, train_labels, 'Train')

    def train(self, train_dataset, train_labels, rbm_batch_sizes, rbm_num_steps,
              ft_batch_size, ft_num_steps, ft_keep_prob=0.72):
        self.run_pretrain(train_dataset, train_labels, rbm_batch_sizes, rbm_num_steps)
        self.unrolling()
        self.run_fine_tuning(train_dataset, train_labels, ft_batch_size, ft_num_steps, ft_keep_prob)