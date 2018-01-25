from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from utils import get_batch
# from time import localtime, strftime


class Autoencoder(object):
    def __init__(self, feature_size, encode_size, dirname,
                 optimizer=tf.train.AdamOptimizer,
                 transfer_func=tf.nn.softplus, sparsity=.0,
                 sparsity_weight=.0, mask_fraction=.0,
                 init_lr=0.1, decay_steps=100.0):
        self.feature_size = feature_size
        self.encode_size = encode_size
        self.transfer_func = transfer_func

        # for sparse autoencoder
        self.sparsity = sparsity
        self.sparsity_weight = sparsity_weight

        # for masking-noise autoencoder
        self.mask_fraction = mask_fraction
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # model
        self.x = tf.placeholder(tf.float32, [None, self.feature_size])
        global_step = tf.Variable(0)
        self.lr = tf.train.exponential_decay(init_lr, global_step,
                                             decay_steps, 0.96,
                                             staircase=False)

        self.train_loss_record = tf.placeholder(tf.float32, name='train_loss')
        self.valid_loss_record = tf.placeholder(tf.float32, name='valid_loss')
        self.kl_divergence_record = tf.placeholder(tf.float32,
                                                   name='kl_divergence')

        self.weights = self._initialize_weights()
        self.encode = self._create_encode_node()
        self.reconstruction = self._create_decode_node()
        self.reconstruction_loss = self._create_reconstruction_loss_node()
        self.kl_divergence = self._create_kl_node()

        self.loss = self._create_loss_node()
        self.optimizer = optimizer(self.lr).minimize(self.loss,
                                                     global_step=global_step)
        """
        time_str = strftime("%b-%d-%Y-%H-%M-%S", localtime())
        self.dirname = self.name + '/Run-' + time_str
        self.train_writer = tf.summary.FileWriter(self.dirname)
        self._create_summaries()
        self.merged_summary = tf.summary.merge_all()
        """
        self.dirname = dirname
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        print('Model built and initialized at %s' % self.dirname)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(
            tf.random_normal([self.feature_size, self.encode_size],
                             mean=0.0, stddev=0.01))
        all_weights['b1'] = tf.Variable(tf.zeros([self.encode_size],
                                                 dtype=tf.float32))

        all_weights['w2'] = tf.Variable(
            tf.random_normal([self.encode_size, self.feature_size],
                             mean=0.0, stddev=0.01))
        all_weights['b2'] = tf.Variable(tf.zeros([self.feature_size],
                                                 dtype=tf.float32))
        return all_weights

    def _create_encode_node(self):
        logits = tf.add(tf.matmul(self.x, self.weights['w1']),
                        self.weights['b1'])
        return self.transfer_func(logits)

    def _create_decode_node(self):
        return tf.add(tf.matmul(self.encode, self.weights['w2']),
                      self.weights['b2'])

    def _create_reconstruction_loss_node(self):
        reconstruction_loss = .5 * tf.reduce_mean(tf.pow(
            tf.subtract(self.reconstruction, self.x), 2.0))
        return reconstruction_loss

    def _create_kl_node(self):
        """Override me if dataset is binary and
        cost is measured by KL-divergence"""
        return None

    def _create_loss_node(self):
        """Override me if you want to add sparsity or regularizations"""
        return self.reconstruction_loss

    def _create_summaries(self):
        print("Start visualize features of encoder layer")
        layer_weight = tf.transpose(self.weights['w1'])
        x_min = tf.reduce_min(layer_weight)
        x_max = tf.reduce_max(layer_weight)
        normalized_layer_weight = tf.div(layer_weight - x_min, x_max - x_min)

        num_neurons = normalized_layer_weight.get_shape()[1].value
        input_dim = normalized_layer_weight.get_shape()[0].value
        print("%d * %d matrix" % (num_neurons, input_dim))
        images = tf.reshape(normalized_layer_weight,
                            [num_neurons, input_dim, 1, 1])
        tf.summary.image('layer1', images, max_outputs=16)

        tf.summary.histogram('layer1 weights', layer_weight)
        tf.summary.histogram('activity', self.encode)
        tf.summary.scalar('max activity', tf.reduce_max(self.encode))
        tf.summary.scalar('min activity', tf.reduce_min(self.encode))
        # tf.summary.scalar('min weight in layer1', x_min)
        # tf.summary.scalar('max weight in layer1', x_max)
        # mean = tf.reduce_mean(layer_weight)
        # tf.summary.scalar('mean in layer1', mean)
        # stddev = tf.sqrt(tf.reduce_mean(tf.square(layer_weight - mean)))
        # tf.summary.scalar('stddev in layer1', stddev)
        tf.summary.scalar('train reconstruction loss', self.train_loss_record)
        tf.summary.scalar('valid reconstruction loss', self.valid_loss_record)
        tf.summary.scalar('kl divergence', self.kl_divergence_record)
        tf.summary.scalar('learning rate', self.lr)
        tf.summary.scalar('encoder size', self.encode_size)
        if isinstance(self, SparseAutoencoder):
            tf.summary.scalar('sparsity', self.sparsity)
            tf.summary.scalar('sparsity weight', self.sparsity_weight)
        if isinstance(self, MaskNoiseAutoencoder):
            tf.summary.scalar('mask fraction', self.mask_fraction)

    def partial_fit(self, X):
        opt, loss, lr = self.sess.run([self.optimizer, self.loss, self.lr],
                                      feed_dict={self.x: X})
        return loss, lr

    def calc_kl_divergence(self, X):
        if self.kl_divergence is not None:
            kl = self.sess.run(self.kl_divergence, feed_dict={self.x: X})
            return kl * self.sparsity_weight
        else:
            return -1.0

    def encode_dataset(self, X):
        return self.sess.run(self.encode, feed_dict={self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def calc_reconstruct_loss(self, X):
        return self.sess.run(self.reconstruction_loss, feed_dict={self.x: X})

    def get_encode_weights(self):
        return self.sess.run(self.weights['w1'])

    def get_encode_biases(self):
        return self.sess.run(self.weights['b1'])

    def train(self, train_dataset, batch_size, num_steps):
        display_step = num_steps // 10
        print('Training for %d steps' % num_steps)

        for step in range(int(num_steps)):
            offset = (batch_size * step) % \
                (train_dataset.shape[0] - batch_size)
            end = (offset + batch_size) % train_dataset.shape[0]
            if end < offset:
                batch_data = np.concatenate((train_dataset[offset:, :],
                                             train_dataset[:end, :]), axis=0)
            else:
                batch_data = train_dataset[offset:(offset + batch_size), :]

            loss = self.partial_fit(batch_data)
            if step % display_step == 0:
                kl = self.calc_kl_divergence(batch_data)
                print("Minibatch(%d cases) loss at step %d: %f(kl=%f)"
                      % (batch_data.shape[0], step, loss, kl))
                batch_loss = self.calc_reconstruct_loss(batch_data)
                print("Batch reconstruction loss: %f" % batch_loss)

        print('%s Training Summary' % self.name)
        train_loss = self.calc_reconstruct_loss(train_dataset)
        print("Trainset reconstruction loss: %f" % train_loss)

    def train_with_labels(self, train_dataset, train_labels, batch_size,
                          num_steps, valid_dataset):
        display_step = num_steps // 10
        # summary_step = num_steps // 100
        num_labels = train_labels.shape[1]
        print('Training for %d steps' % num_steps)

        y = np.argmax(train_labels, 1)
        X = np.array([train_dataset[y == i, :] for i in range(num_labels)])
        Y = np.array([train_labels[y == i, :] for i in range(num_labels)])

        batch_size /= num_labels
        for step in range(num_steps):
            if num_labels > 2:
                train0, label0 = get_batch(X[0], Y[0], step, batch_size)
                train1, label1 = get_batch(X[1], Y[1], step, batch_size)
                train2, label2 = get_batch(X[2], Y[2], step, batch_size)
                # train3 = X[3][np.random.choice(X[3].shape[0], 50), :]
                train3, label3 = get_batch(X[3], Y[3], step, batch_size)
                train4, label4 = get_batch(X[4], Y[4], step, batch_size)
                batch_data = np.concatenate(
                    (train0, train1, train2, train3, train4), axis=0)
            else:
                train0, label0 = get_batch(X[0], Y[0], step, batch_size)
                train1, label1 = get_batch(X[1], Y[1], step, batch_size)
                batch_data = np.concatenate((train0, train1), axis=0)

            perm = np.random.permutation(batch_data.shape[0])
            batch_data = batch_data[perm, :]

            loss, lr = self.partial_fit(batch_data)
            """
            if step != 0 and step % summary_step == 0:
                kl = self.calc_kl_divergence(batch_data)
                train_loss = self.calc_reconstruct_loss(train_dataset)
                valid_loss = self.calc_reconstruct_loss(valid_dataset)
                summaries = self.sess.run(
                    self.merged_summary,
                    feed_dict={self.x: batch_data,
                               self.train_loss_record: train_loss,
                               self.valid_loss_record: valid_loss,
                               self.kl_divergence_record: kl,
                               self.keep_prob: 1.0 - self.mask_fraction})
                self.train_writer.add_summary(summaries, step)
            """
            if step != 0 and step % display_step == 0:
                kl = self.calc_kl_divergence(batch_data)
                batch_loss = self.calc_reconstruct_loss(batch_data)
                train_loss = self.calc_reconstruct_loss(train_dataset)
                valid_loss = self.calc_reconstruct_loss(valid_dataset)
                print("Minibatch(%d cases) loss at step %d: %.6f"
                      % (batch_data.shape[0], step, loss))
                print("kl=%.4f, lr=%s" % (kl, lr))
                print("Batch reconstruction loss: %f" % batch_loss)
                print("Train reconstruction loss: %f" % train_loss)
                print("Valid reconstruction loss: %f" % valid_loss)

        print("Finish training")


class SparseAutoencoder(Autoencoder):
    def __init__(self, feature_size, encode_size, dirname,
                 optimizer=tf.train.AdamOptimizer,
                 transfer_func=tf.nn.sigmoid,
                 sparsity=0.01, sparsity_weight=0.05,
                 init_lr=0.1, decay_steps=10000):
        super(SparseAutoencoder, self).__init__(
            feature_size, encode_size, dirname, optimizer,
            transfer_func, sparsity, sparsity_weight,
            init_lr=init_lr, decay_steps=decay_steps)

    def _create_kl_node(self):
        """Calculate the kl divergence between active nodes
        and sparsity vector
        return tf.reduce_sum(tf.reduce_mean(tf.abs(self.encode),
                                            reduction_indices=0))
        """
        sparsity_vector = tf.constant(self.sparsity, shape=[self.encode_size],
                                      dtype=tf.float32, name='sparsity_vector')
        # convert average activity to probabilistic distribution
        activity = tf.reduce_mean(self.encode, axis=0)
        # activity = tf.reshape(activity, [self.encode_size])
        """
        # KL(P, Q) = cross_entropy(P, Q) - entropy(P),
        # where P is sparsity vector
        cross_entropy = tf.reduce_sum(tf.multiply(sparsity_vector,
                                                  -tf.log(activity)))
        entropy = tf.reduce_sum(
            tf.multiply(sparsity_vector,
                        -tf.log(tf.transpose(sparsity_vector))))
        return tf.subtract(cross_entropy, entropy)
        """
        logdiv = tf.log(tf.div(sparsity_vector, activity))
        return tf.reduce_sum(tf.multiply(sparsity_vector, logdiv))

    def _create_loss_node(self):
        return tf.add(self.reconstruction_loss,
                      self.kl_divergence * self.sparsity_weight)


class MaskNoiseAutoencoder(Autoencoder):
    def __init__(self, feature_size, encode_size, dirname,
                 optimizer=tf.train.AdamOptimizer,
                 transfer_func=tf.nn.sigmoid,
                 mask_fraction=0.4):
        super(MaskNoiseAutoencoder, self).__init__(
            feature_size, encode_size, dirname, optimizer, transfer_func,
            mask_fraction=mask_fraction)

    def _create_encode_node(self):
        mask = tf.multiply(self.keep_prob,
                           tf.nn.dropout(self.x, self.keep_prob))
        logits = tf.add(tf.matmul(mask, self.weights['w1']),
                        self.weights['b1'])
        return self.transfer_func(logits)

    def partial_fit(self, X, lr):
        opt, loss = self.sess.run(
            [self.optimizer, self.loss],
            feed_dict={self.x: X, self.encode_lr: lr,
                       self.keep_prob: 1.0 - self.mask_fraction})
        return loss

    def calc_reconstruct_loss(self, X):
        return self.sess.run(self.loss,
                             feed_dict={self.x: X, self.keep_prob: 1.0})

    def encode_dataset(self, X):
        return self.sess.run(self.encode,
                             feed_dict={self.x: X, self.keep_prob: 1.0})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction,
                             feed_dict={self.x: X, self.keep_prob: 1.0})
