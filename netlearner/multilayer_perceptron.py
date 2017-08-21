from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils import xavier_init, accuracy, measure_prediction, get_batch
from time import localtime, strftime


class MultilayerPerceptron(object):
    def __init__(self, feature_size, hidden_layer_sizes, num_labels,
                 trans_func=tf.nn.relu, reg_func=tf.nn.l2_loss, beta=0.009,
                 optimizer=tf.train.GradientDescentOptimizer, name='mlp'):
        self.feature_size = feature_size
        self.layer_sizes = hidden_layer_sizes + [num_labels]
        self.num_labels = num_labels
        self.trans_func = trans_func

        self.params = self._create_variables()
        self.x = tf.placeholder(tf.float32, [None, feature_size], name='input')
        self.t = tf.placeholder(tf.float32, [None, num_labels], name='target')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='lr')
        self.train_accuracy_record = tf.placeholder(tf.float32, name='train_accu')
        self.valid_loss_record = tf.placeholder(tf.float32, name='valid_loss')
        self.valid_accuracy_record = tf.placeholder(tf.float32, name='valid_accu')
        self.test_accuracy_record = tf.placeholder(tf.float32, name='test_accu')

        self.final_logits = self._create_forward()
        self.classify_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.final_logits,
                                                    labels=self.t))
        self.regterm = self._create_regterm(reg_func, beta)

        self.loss = self.classify_loss + self.regterm
        self.optimizer = optimizer(learning_rate=self.learning_rate).minimize(self.loss)

        self.predict = tf.nn.softmax(self.final_logits)

        time_str = strftime("%b-%d-%Y-%H-%M-%S", localtime())
        self.dirname = name + '/Run-' + time_str
        self.train_writer = tf.summary.FileWriter(self.dirname)
        # self._create_summaries()
        # self.merged_summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()
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
            if i < len(self.layer_sizes) - 1:
                activity = self.trans_func(logits)
                next_input = tf.nn.dropout(activity, self.keep_prob)
            else:
                next_input = logits  # tf.nn.sigmoid(logits)
                print('Last layer is softmax(logits)')

        return next_input

    def _create_regterm(self, reg_func, beta):
        regterm = reg_func([0.0])
        for (i, hsize) in enumerate(self.layer_sizes):
            regterm = tf.add(regterm, reg_func(self.params['w%d' % i]))
        return tf.multiply(beta, regterm)

    def _create_summaries(self):
        tf.summary.scalar('input dimension', self.feature_size)
        for (layer, size) in enumerate(self.layer_sizes):
            print("Start visualize features on layer %d" % (layer + 1))
            layer_weight = tf.transpose(self.params['w%d' % layer])
            x_min = tf.reduce_min(layer_weight)
            x_max = tf.reduce_max(layer_weight)
            normalized_layer_weight = tf.div(layer_weight - x_min, x_max - x_min)

            num_neurons = normalized_layer_weight.get_shape()[0].value
            input_dim = normalized_layer_weight.get_shape()[1].value
            print("%d * %d matrix" % (num_neurons, input_dim))
            images = tf.reshape(normalized_layer_weight, [num_neurons, input_dim, 1, 1])
            tf.summary.image('layer%d' % (layer + 1), images, max_outputs=16)

            tf.summary.scalar('layer %d size' % (layer + 1), size)
            tf.summary.histogram('histogram of layer %d weights' % (layer + 1), layer_weight)
            # tf.summary.scalar('min weight in layer %d' % (layer + 1), x_min)
            # tf.summary.scalar('max weight in layer %d' % (layer + 1), x_max)
            # mean = tf.reduce_mean(layer_weight)
            # tf.summary.scalar('mean in layer %d' % (layer + 1), mean)
            # stddev = tf.sqrt(tf.reduce_mean(tf.square(layer_weight - mean)))
            # tf.summary.scalar('stddev in layer %d' % (layer + 1), stddev)

        tf.summary.scalar('learning rate', self.learning_rate)
        tf.summary.scalar('dropout keep probability', self.keep_prob)
        # tf.summary.scalar('regularization loss', self.regterm_record)
        tf.summary.scalar('train accuracy', self.train_accuracy_record)
        # tf.summary.scalar('valid loss', self.valid_loss_record)
        tf.summary.scalar('valid accuracy', self.valid_accuracy_record)
        tf.summary.scalar('test accuracy', self.test_accuracy_record)

    def fit(self, X, T, lr, prob=0.5):
        opt, loss, reg = self.sess.run([self.optimizer, self.loss, self.regterm],
                                       feed_dict={self.x: X, self.t: T,
                                                  self.keep_prob: prob,
                                                  self.learning_rate: lr})
        return loss, reg

    def make_prediction(self, X):
        # Don't use dropout during testing
        return self.sess.run(self.predict,
                             feed_dict={self.x: X,
                                        self.keep_prob: 1.0})

    def calc_total_loss(self, X, T, prob=1.0):
        loss = self.sess.run(self.loss,
                             feed_dict={self.x: X, self.t: T,
                                        self.keep_prob: prob})
        return loss

    def make_summary(self, train_accu, valid_loss, valid_accu, test_accu, lr, keep_prob, step):
        summaries = self.sess.run(self.merged_summary,
                                  feed_dict={self.keep_prob: keep_prob,
                                             self.train_accuracy_record: train_accu,
                                             self.valid_loss_record: valid_loss,
                                             self.valid_accuracy_record: valid_accu,
                                             self.test_accuracy_record: test_accu,
                                             self.learning_rate: lr})
        self.train_writer.add_summary(summaries, step)

    def get_weights(self, name='w0'):
        return self.sess.run(self.params[name])

    def exit(self):
        self.sess.close()

    def train_with_labels(self, train_dataset, train_labels, batch_size,
                          num_steps, init_lr,
                          valid_dataset, valid_labels,
                          test_dataset, test_labels, keep_prob=0.8):
        display_step = num_steps // 10
        summary_step = num_steps // 100
        print('Training for %d steps' % num_steps)

        train_perm = np.random.permutation(train_dataset.shape[0])
        train_dataset = train_dataset[train_perm, :]
        train_labels = train_labels[train_perm, :]

        y = np.argmax(train_labels, 1)
        X = np.array([train_dataset[y == i, :] for i in range(self.num_labels)])
        Y = np.array([train_labels[y == i, :] for i in range(self.num_labels)])

        train_accu = 0.0
        valid_accu = 0.0
        valid_loss = 0.0
        lr = init_lr
        batch_size /= self.num_labels
        for step in range(num_steps):
            if self.num_labels > 2:
                train0, label0 = get_batch(X[0], Y[0], step, batch_size)
                train1, label1 = get_batch(X[1], Y[1], step, batch_size)
                train2, label2 = get_batch(X[2], Y[2], step, batch_size)
                # train3 = X[3][np.random.choice(X[3].shape[0], 50), :]
                if batch_size < X[3].shape[0]:
                    train3, label3 = get_batch(X[3], Y[3], step, batch_size)
                else:
                    train3, label3 = X[3][:, :], Y[3][:, :]
                train4, label4 = get_batch(X[4], Y[4], step, batch_size)
                batch_data = np.concatenate((train0, train1, train2, train3, train4), axis=0)
                batch_labels = np.concatenate((label0, label1, label2, label3, label4), axis=0)
            else:
                train0, label0 = get_batch(X[0], Y[0], step, batch_size)
                train1, label1 = get_batch(X[1], Y[1], step, batch_size)
                batch_data = np.concatenate((train0, train1), axis=0)
                batch_labels = np.concatenate((label0, label1), axis=0)

            perm = np.random.permutation(batch_data.shape[0])
            batch_data = batch_data[perm, :]
            batch_labels = batch_labels[perm, :]
            lr = init_lr * np.power(0.32, float(step) / float(num_steps))
            loss, reg = self.fit(batch_data, batch_labels, lr, keep_prob)

            if step != 0 and step % summary_step == 0:
                train_predict = self.make_prediction(train_dataset)
                train_accu = accuracy(train_predict, train_labels)
                valid_predict = self.make_prediction(valid_dataset)
                valid_loss, valid_reg = self.sess.run([self.loss, self.regterm],
                                                      feed_dict={self.x: valid_dataset,
                                                                 self.t: valid_labels,
                                                                 self.keep_prob: keep_prob,
                                                                 self.learning_rate: lr})
                valid_accu = accuracy(valid_predict, valid_labels)
                test_predict = self.make_prediction(test_dataset)
                test_accu = accuracy(test_predict, test_labels)
                # self.make_summary(train_accu, valid_loss, valid_accu, test_accu,
                                  # lr, keep_prob, stpes)

            if step != 0 and step % display_step == 0:
                batch_predict = self.make_prediction(batch_data)
                valid_predict = self.make_prediction(valid_dataset)
                print("Minibatch(%d cases) loss at step %d: %.6f(regterm=%.4f, lr=%.6f)"
                      % (batch_labels.shape[0], step, loss, reg, lr))
                print("Minibatch train accuracy: %f%%" % accuracy(batch_predict, batch_labels))
                print("Validation accuracy: %f%%" % accuracy(valid_predict, valid_labels))

        print('Multilayer Perceptron trained')

        train_loss = self.calc_total_loss(train_dataset, train_labels)
        print("Trainset total loss: %f" % train_loss)
        train_predict = self.make_prediction(train_dataset)
        train_accu = accuracy(train_predict, train_labels)
        measure_prediction(train_predict, train_labels, self.dirname, 'Train')

        valid_predict = self.make_prediction(valid_dataset)
        valid_accu = accuracy(valid_predict, valid_labels)
        measure_prediction(valid_predict, valid_labels, self.dirname, 'Valid')

        test_predict = self.make_prediction(test_dataset)
        test_accu = accuracy(test_predict, test_labels)
        measure_prediction(test_predict, test_labels, self.dirname, 'Test')

        # self.make_summary(train_accu, valid_loss, valid_accu, test_accu,
                          # lr, keep_prob, num_stpes)
