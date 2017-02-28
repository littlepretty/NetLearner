from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils import xavier_init, accuracy, measure_prediction, get_batch


class MultilayerPerceptron(object):
    def __init__(self, feature_size, hidden_layer_sizes, num_labels,
                 init_learning_rate=0.64, decay_steps=10000, decay_base=0.96,
                 trans_func=tf.nn.relu, reg_func=tf.nn.l2_loss, beta=0.009,
                 optimizer=tf.train.GradientDescentOptimizer):
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
        learning_rate = tf.train.exponential_decay(
            init_learning_rate, global_step, decay_steps,
            decay_base, staircase=True)
        self.optimizer = optimizer(learning_rate).minimize(self.loss)

        self.predict = tf.nn.softmax(self.final_logits)

        self.train_writer = tf.summary.FileWriter('mlp/train')
        self._create_summaries()
        self.merged_summary = tf.summary.merge_all()

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
                next_input = logits
                print('Last layer is just linear logistic')

        return next_input

    def _create_regterm(self, reg_func, beta):
        regterm = reg_func([0.0])
        for (i, hsize) in enumerate(self.layer_sizes):
            regterm = tf.add(regterm, reg_func(self.params['w%d' % i]))
        return tf.mul(beta, regterm)

    def fit(self, X, T, prob=0.5):
        opt, loss, reg = self.sess.run(
            [self.optimizer, self.loss, self.regterm],
            feed_dict={self.x: X, self.t: T, self.keep_prob: prob})
        return loss, reg

    def make_prediction(self, X, prob=1.0):
        return self.sess.run(self.predict,
                             feed_dict={self.x: X,
                                        self.keep_prob: prob})

    def calc_total_loss(self, X, T, prob=1.0):
        loss = self.sess.run(self.loss,
                             feed_dict={self.x: X, self.t: T,
                                        self.keep_prob: prob})
        return loss

    def train(self, train_dataset, train_labels, batch_size, num_steps, keep_prob=0.5):
        display_step = num_steps / 10
        print('Training for %d steps' % num_steps)

        for step in range(num_steps):
            offset = (batch_size * step) % train_labels.shape[0]
            end = (offset + batch_size) % train_labels.shape[0]
            if end < offset:
                batch_data = np.concatenate((train_dataset[offset:, :],
                                             train_dataset[:end, :]), axis=0)
                batch_labels = np.concatenate((train_labels[offset:, :],
                                               train_labels[:end, :]), axis=0)
            else:
                batch_data = train_dataset[offset:(offset + batch_size), :]
                batch_labels = train_labels[offset:(offset + batch_size), :]

            loss, reg = self.fit(batch_data, batch_labels, keep_prob)
            if step % display_step == 0:
                print("Minibatch %d-%d loss at step %d:\t%f(regterm=%f)"
                      % (offset, end, step, loss, reg))
                batch_predict = self.make_prediction(batch_data)
                print("Minibatch train accuracy: %f%%" %
                      accuracy(batch_predict, batch_labels))

        print('Multilayer Perceptron trained')
        train_loss = self.calc_total_loss(train_dataset, train_labels)
        train_predict = self.make_prediction(train_dataset)
        print("Trainset total loss: %f" % train_loss)
        measure_prediction(train_predict, train_labels, 'Train')

    def train_with_bias(self, train_dataset, train_labels, batch_size,
                        num_steps, keep_prob=0.5):
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
                    print("Minibatch(%d cases) loss at step %d:\t%f(regterm=%f)"
                          % (batch_labels.shape[0], step, loss, reg))
                    batch_predict = self.make_prediction(batch_data)
                    print("Minibatch train accuracy: %f%%" %
                          accuracy(batch_predict, batch_labels))

        print('Multilayer Perceptron trained')
        train_loss = self.calc_total_loss(train_dataset, train_labels)
        train_predict = self.make_prediction(train_dataset)
        print("Trainset total loss: %f" % train_loss)
        measure_prediction(train_predict, train_labels, 'Train')
        self.train_writer.add_summary(self.sess.run(self.merged_summary))

    def _create_summaries(self):
        for (layer, _) in enumerate(self.layer_sizes):
            print("Start visualize features on layer %d" % (layer + 1))
            layer_weight = tf.transpose(self.params['w%d' % layer])
            x_min = tf.reduce_min(layer_weight)
            x_max = tf.reduce_max(layer_weight)
            normalized_layer_weight = tf.div(layer_weight - x_min, x_max - x_min)

            num_images = normalized_layer_weight.get_shape()[0].value
            image_size = normalized_layer_weight.get_shape()[1].value
            print("%d * %d matrix" % (num_images, image_size))
            edge = int(np.sqrt(image_size))
            images = tf.reshape(normalized_layer_weight, [num_images, edge, edge, 1])
            tf.image_summary('layer%d' % (layer + 1), images, max_images=num_images)
            tf.summary.histogram('histogram of layer %d weights' % (layer + 1), layer_weight)
            tf.summary.scalar('min weight in layer %d' % (layer + 1), x_min)
            tf.summary.scalar('max weight in layer %d' % (layer + 1), x_max)
            mean = tf.reduce_mean(layer_weight)
            tf.summary.scalar('mean in layer %d' % (layer + 1), mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(layer_weight - mean)))
            tf.summary.scalar('stddev in layer %d' % (layer + 1), stddev)
