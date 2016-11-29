from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils import xavier_init, sample_prob_dist, get_batch


class RestrictedBoltzmannMachine(object):
    def __init__(self, num_visible, num_hidden, batch_size,
                 lr=0.1, trans_func=tf.nn.sigmoid):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.lr = lr
        self.trans_func = trans_func

        self.weights = self._initialize_weights()

        # v is used for both cd and generate hidden states
        self.v = tf.placeholder(tf.float32, [None, num_visible], name='v')
        # self.h = tf.placeholder(tf.float32, [None, num_hidden], name='h')

        self.vrand = tf.placeholder(tf.float32, [None, num_visible], name='vrand')
        self.hrand = tf.placeholder(tf.float32, [None, num_hidden], name='hrand')

        self.batch_size = batch_size

        # generate hidden state from visible state x
        # and random distribution hrand
        self.encode = self.sample_hidden_from_visible(self.v)[0]

        # generate visible state from hidden state x
        # and random distribution vrand
        self.reconstruct = self.sample_visible_from_hidden(self.encode)[0]

        # training algorithm: constractive divergence
        self.updates = self._create_cd1()
        self.loss = self._create_reconstruct_loss()

        # measure the goodness of the weights
        self.goodness = self._create_goodness()

        init = tf.initialize_all_variables()

        self.sess = tf.Session()
        self.sess.run(init)
        print('Initialized')

    def _initialize_weights(self):
        weights = dict()
        weights['w'] = tf.Variable(xavier_init(self.num_visible,
                                               self.num_hidden))
        weights['bh'] = tf.Variable(tf.zeros(shape=[self.num_hidden],
                                             dtype=tf.float32))
        weights['bv'] = tf.Variable(tf.zeros(shape=[self.num_visible],
                                             dtype=tf.float32))
        return weights

    def _create_goodness(self):
        association = tf.matmul(tf.transpose(self.v), self.encode)
        neg_energy = tf.reduce_mean(tf.mul(self.weights['w'], association))
        return neg_energy

    def sample_visible_from_hidden(self, hidden):
        logit = tf.matmul(hidden, tf.transpose(self.weights['w']))
        vprob = self.trans_func(logit)
        vstate = sample_prob_dist(vprob, self.vrand)
        return vprob, vstate

    def sample_hidden_from_visible(self, visible):
        logit = tf.matmul(visible, self.weights['w'])
        hprob = self.trans_func(logit)
        hstate = sample_prob_dist(hprob, self.hrand)
        return hprob, hstate

    def gibbs_sampling_step(self, visible):
        # postive phase
        pos_hprob, pos_hstate = self.sample_hidden_from_visible(visible)

        # negative phase
        neg_vprob, neg_vstate = self.sample_visible_from_hidden(pos_hprob)
        neg_hprob, neg_hstate = self.sample_hidden_from_visible(neg_vprob)
        return [pos_hprob, pos_hstate,
                neg_vprob, neg_vstate,
                neg_hprob, neg_hstate]

    def _create_cd1(self):
        [pos_hprob, pos_hstate, neg_vprob,
         neg_vstate, neg_hprob, neg_hstate] = self.gibbs_sampling_step(self.v)

        pos_association = tf.matmul(tf.transpose(self.v), pos_hstate)
        neg_association = tf.matmul(tf.transpose(neg_vstate), neg_hstate)

        gradient_w = self.lr * tf.sub(pos_association,
                                      neg_association) / self.batch_size
        update_w = tf.assign_add(self.weights['w'], gradient_w)

        # g_bh = self.lr * tf.reduce_mean(tf.sub(pos_hprob, neg_hprob), 0)
        # update_bh = tf.assign_add(self.weights['bh'], g_bh)

        # g_bv = self.lr * tf.reduce_mean(tf.sub(self.v, neg_vprob), 0)
        # update_bv = tf.assign_add(self.weights['bv'], g_bv)

        # mean squared error
        loss = 0.5 * tf.reduce_sum(tf.square(tf.sub(neg_vprob, self.v)))

        return [loss, update_w]
        # return [loss, update_w, update_bh, update_bv]

    def _create_reconstruct_loss(self):
        loss = 0.5 * tf.reduce_sum(tf.square(tf.sub(self.reconstruct, self.v)))
        return loss

    def run_train_step(self, V):
        Vrand = np.random.random([V.shape[0], self.num_visible])
        Hrand = np.random.random((V.shape[0], self.num_hidden))
        return self.sess.run(self.updates,
                             feed_dict={self.v: V, self.vrand: Vrand, self.hrand: Hrand})

    def cal_goodness(self, V, Hrand):
        return self.sess.run(self.goodness,
                             feed_dict={self.v: V, self.hrand: Hrand})

    def encode_dataset(self, V, Hrand):
        return self.sess.run(self.encode,
                             feed_dict={self.v: V, self.hrand: Hrand})

    def reconstruct_dataset(self, V, Vrand, Hrand):
        return self.sess.run(self.reconstruct,
                             feed_dict={self.v: V,
                                        self.hrand: Hrand,
                                        self.vrand: Vrand})

    def calc_reconstruct_loss(self, V):
        vrand = np.random.random(size=(V.shape[0], self.num_visible))
        hrand = np.random.random(size=(V.shape[0], self.num_hidden))
        return self.sess.run(self.loss,
                             feed_dict={self.v: V, self.vrand: vrand,
                                        self.hrand: hrand})

    def train(self, train_dataset, batch_size, num_steps):
        display_steps = num_steps / 10
        for step in range(num_steps):
            offset = (batch_size * step) % (train_dataset.shape[0] - batch_size)
            end = (offset + batch_size) % train_dataset.shape[0]
            if end < offset:
                batch_data = np.concatenate((train_dataset[offset:, :],
                                             train_dataset[:end, :]), axis=0)
            else:
                batch_data = train_dataset[offset:(offset + batch_size), :]

            l, _ = self.run_train_step(batch_data)
            if step % display_steps == 0:
                batch_loss = self.calc_reconstruct_loss(batch_data)
                print("Batch loss at step %d: %f" % (step, l))
                print("Batch reconstruction loss at step %d: %f" % (step, batch_loss))

        print('Restricted Boltzmann Machine trained')
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
            l, _ = self.run_train_step(batch_data)

            if step % display_step == 0:
                batch_loss = self.calc_reconstruct_loss(batch_data)
                print("Batch loss at step %d: %f" % (step, l))
                print("Batch reconstruction loss at step %d: %f" % (step, batch_loss))

        print('Restricted Boltzmann Machine trained')
        train_loss = self.calc_reconstruct_loss(train_dataset)
        print("Trainset reconstruction loss: %f" % train_loss)

    def get_weights(self, name):
        return self.sess.run(self.weights[name])