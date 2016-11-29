from __future__ import print_function
import numpy as np
import tensorflow as tf
from utils import xavier_init, get_batch


class Autoencoder(object):
    def __init__(self, feature_size, encode_size, encode_lr=0.001,
                 l2_weight=0.001, sparsity_weight=0.05,
                 transfer_func=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer,
                 name='Autoencoder'):
        self.feature_size = feature_size
        self.encode_size = encode_size
        self.l2_weight = l2_weight
        self.sparsity_weight = sparsity_weight

        self.transfer_func = transfer_func
        self.name = name

        # model
        self.x = tf.placeholder(tf.float32, [None, self.feature_size])
        self.keep_prob = tf.placeholder(tf.float32)
        self.sparsity_vector = tf.placeholder(tf.float32, [self.encode_size])

        self.weights = self._initialize_weights()

        self.encode = self._create_encode_node()
        self.reconstruction = self._create_reconstruction_node()

        self.reconstruction_loss = self._create_reconstruction_loss_node()
        self.regterm = self._create_regularization_node()
        self.kl_divergence = self._create_kl_node()

        self.loss = self._create_loss_node()
        self.optimizer = optimizer(encode_lr).minimize(self.loss)

        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
        print('%s built and initialized' % name)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.feature_size, self.encode_size))
        all_weights['b1'] = tf.Variable(tf.zeros([self.encode_size], dtype=tf.float32))

        all_weights['w2'] = tf.Variable(tf.zeros([self.encode_size, self.feature_size],
                                                 dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.feature_size], dtype=tf.float32))
        return all_weights

    def _create_encode_node(self):
        logits = tf.add(tf.matmul(self.x, self.weights['w1']),
                        self.weights['b1'])
        return self.transfer_func(logits)

    def _create_reconstruction_node(self):
        return tf.add(tf.matmul(self.encode, self.weights['w2']),
                      self.weights['b2'])

    def _create_regularization_node(self):
        l2regterm = tf.nn.l2_loss(self.weights['w1'])
        l2regterm = tf.add(l2regterm, tf.nn.l2_loss(self.weights['w2']))
        return l2regterm

    def _create_reconstruction_loss_node(self):
        reconstruction_loss = 0.5 * tf.reduce_sum(tf.pow(
            tf.sub(self.reconstruction, self.x), 2.0))
        return reconstruction_loss

    def _create_kl_node(self):
        return None

    def _create_loss_node(self):
        return self.reconstruction_loss + self.regterm

    def partial_fit(self, X):
        opt, loss = self.sess.run([self.optimizer, self.loss],
                                  feed_dict={self.x: X})
        return loss

    def calc_regularization(self):
        return self.l2_weight * self.sess.run(self.regterm)

    def calc_reconstruct_loss(self, X):
        return self.sess.run(self.reconstruction_loss, feed_dict={self.x: X})

    def calc_kl_divergence(self, X):
        return 0

    def encode_dataset(self, X):
        return self.sess.run(self.encode, feed_dict={self.x: X})

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
            end = (offset + batch_size) % train_dataset.shape[0]
            if end < offset:
                batch_data = np.concatenate((train_dataset[offset:, :],
                                             train_dataset[:end, :]), axis=0)
            else:
                batch_data = train_dataset[offset:(offset + batch_size), :]

            loss = self.partial_fit(batch_data)
            if step % display_step == 0:
                reg = self.calc_regularization()
                if self.kl_divergence is not None:
                    kl = self.calc_kl_divergence(batch_data)
                    print("Minibatch(%d cases) loss at step %d: %f(kl=%f, regterm=%f)"
                          % (batch_data.shape[0], step, loss, kl, reg))
                else:
                    print("Minibatch(%d cases) loss at step %d: %f(regterm=%f)"
                          % (batch_data.shape[0], step, loss, reg))
                batch_loss = self.calc_reconstruct_loss(batch_data)
                print("Batch reconstruction loss: %f" % batch_loss)

        print('%s Training Summary' % self.name)
        train_loss = self.calc_reconstruct_loss(train_dataset)
        print("Trainset reconstruction loss: %f" % train_loss)

    def train_with_labels(self, train_dataset, train_labels, batch_size, num_steps):
        display_step = num_steps / 10
        num_labels = train_labels.shape[1]
        print('Training for %d steps' % num_steps)

        y = np.argmax(train_labels, 1)
        X = np.array([train_dataset[y == i, :] for i in range(num_labels)])
        Y = np.array([train_labels[y == i, :] for i in range(num_labels)])
        print(X[0].shape, Y[0].shape)
        print(X[1].shape, X[1].shape)

        batch_size /= (num_labels - 1)
        for step in range(num_steps):
            train0, _ = get_batch(X[0], Y[0], step, batch_size)
            train1, _ = get_batch(X[1], Y[1], step, batch_size)
            train2, _ = get_batch(X[2], Y[2], step, batch_size)
            train3 = X[3][:, :]
            train4, label4 = get_batch(X[4], Y[4], step, batch_size)

            batch_data = np.concatenate((train0, train1, train2, train3, train4), axis=0)
            perm = np.random.permutation(batch_data.shape[0])
            batch_data = batch_data[perm, :]

            loss = self.partial_fit(batch_data)
            if step % display_step == 0:
                reg = self.calc_regularization()
                if self.kl_divergence is not None:
                    kl = self.calc_kl_divergence(batch_data)
                    print("Minibatch(%d cases) loss at step %d: %f(kl=%f, regterm=%f)"
                          % (batch_data.shape[0], step, loss, kl, reg))
                else:
                    print("Minibatch(%d cases) loss at step %d: %f(regterm=%f)"
                          % (batch_data.shape[0], step, loss, reg))
                batch_loss = self.calc_reconstruct_loss(batch_data)
                print("Batch reconstruction loss: %f" % batch_loss)

        print('%s Training Summary' % self.name)
        train_loss = self.calc_reconstruct_loss(train_dataset)
        print("Trainset reconstruction loss: %f" % train_loss)


class SparseAutoencoder(Autoencoder):
    def __init__(self, feature_size, encode_size,
                 sparsity=0.05, sparsity_weight=0.05,
                 encode_lr=0.001, l2_weight=0.003,
                 transfer_func=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer):
        super(SparseAutoencoder, self).__init__(
            feature_size, encode_size, encode_lr,
            l2_weight, sparsity_weight, transfer_func, optimizer, 'Sparse Autoencoder')
        self.sparsity = sparsity

    def _create_kl_node(self):
        return tf.reduce_sum(
            tf.reduce_mean(tf.abs(self.encode),
                           reduction_indices=0))
        # convert average activity to probabilistic distribution
        # activity = tf.reduce_mean(self.encode, reduction_indices=0)
        # activity = tf.div(activity, tf.reduce_sum(self.encode))
        #
        # neg_activity = tf.sub(tf.ones([self.encode_size]), activity)
        # neg_sparsity_vector = tf.sub(tf.ones([self.encode_size]), self.sparsity_vector)
        # # KL(P, Q) = cross_entropy(P, Q) - entropy(P), where P is sparsity vector
        # cross_entropy = tf.mul(self.sparsity_vector, tf.log(activity))
        # cross_entropy = -1 * tf.add(cross_entropy, tf.mul(neg_sparsity_vector, tf.log(neg_activity)))
        # cross_entropy = tf.reduce_sum(cross_entropy)
        #
        # entropy = tf.mul(self.sparsity_vector, tf.log(self.sparsity_vector))
        # entropy = -1 * tf.add(entropy, tf.mul(neg_sparsity_vector, tf.log(neg_sparsity_vector)))
        # entropy = tf.reduce_sum(entropy)
        #
        # kl_divergence = tf.sub(cross_entropy, entropy)
        # return kl_divergence

    def _create_loss_node(self):
        loss = tf.add(self.reconstruction_loss, self.regterm * self.l2_weight)
        loss = tf.add(loss, self.kl_divergence * self.sparsity_weight)
        return loss

    def partial_fit(self, X):
        sparsity_vector = self.sparsity * np.ones([self.encode_size], dtype=float)
        opt, loss = self.sess.run([self.optimizer, self.loss],
                                  feed_dict={self.x: X,
                                             self.sparsity_vector: sparsity_vector})
        return loss

    def calc_kl_divergence(self, X):
        kl = self.sess.run(self.kl_divergence, feed_dict={self.x: X})
        return kl * self.sparsity_weight


class MaskNoiseAutoencoder(Autoencoder):
    def __init__(self, feature_size, encode_size, mask_prob,
                 encode_lr=0.001, l2_weight=0.003, sparsity_weight=0.05,
                 transfer_func=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer):
        super(MaskNoiseAutoencoder, self).__init__(
            feature_size, encode_size, encode_lr,
            l2_weight, sparsity_weight, transfer_func, optimizer, 'Denoise Autoencoder')
        self.mask_prob = mask_prob

    def _create_encode_node(self):
        mask = tf.mul(self.keep_prob, tf.nn.dropout(self.x, self.keep_prob))
        logits = tf.add(tf.matmul(mask, self.weights['w1']), self.weights['b1'])
        return self.transfer_func(logits)

    def partial_fit(self, X):
        opt, loss = self.sess.run([self.optimizer, self.loss],
                                  feed_dict={self.x: X, self.keep_prob: self.mask_prob})
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
