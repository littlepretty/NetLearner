from __future__ import print_function
import tensorflow as tf
from utils import xavier_init, accuracy, measure_prediction


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
            weights['w%d' % i] = tf.Variable(xavier_init(fsize, hsize, 4))
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

    def make_prediction(self, X, prob=1.0):
        return self.sess.run(self.predict,
                             feed_dict={self.x: X,
                                        self.keep_prob: prob})

    def calc_total_loss(self, X, T, prob=1.0):
        return self.sess.run(self.loss,
                             feed_dict={self.x: X, self.t: T,
                                        self.keep_prob: prob})

    def train(self, train_dataset, train_labels, batch_size, num_steps, keep_prob=0.5):
        display_step = num_steps / 10
        print('Training for %d steps' % num_steps)

        for step in range(num_steps):
            offset = (batch_size * step) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            loss, reg = self.fit(batch_data, batch_labels, keep_prob)
            if step % display_step == 0:
                print("Minibatch loss at step %d:\t%f(regterm=%f)"
                      % (step, loss, reg))
                batch_predict = self.make_prediction(batch_data)
                print("Minibatch train accuracy: %f%%" %
                      accuracy(batch_predict, batch_labels))

        print('Multilayer Perceptron trained')
        train_loss = self.calc_total_loss(train_dataset, train_labels)
        train_predict = self.make_prediction(train_dataset)
        train_accuracy = accuracy(train_predict, train_labels)
        print("Trainset total loss: %f" % train_loss)
        print("Trainset total accuracy: %f" % train_accuracy)
        measure_prediction(train_predict, train_labels, 'Train')
