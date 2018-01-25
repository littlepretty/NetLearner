from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils import sample_prob_dist, get_batch
# from time import localtime, strftime


class RestrictedBoltzmannMachine(object):
    def __init__(self, num_visible, num_hidden, batch_size,
                 trans_func=tf.nn.sigmoid, num_labels=5,
                 restore_dir=None, dirname='RBM'):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_labels = num_labels
        self.trans_func = trans_func
        self.dirname = dirname

        self.weights = self._initialize_weights()

        # v is used for both cd and generate hidden states
        self.v = tf.placeholder(tf.float32, [None, num_visible], name='v')
        # self.h = tf.placeholder(tf.float32, [None, num_hidden], name='h')
        self.one = tf.placeholder(tf.float32, [None, num_hidden], name='1fe')

        self.vrand = tf.placeholder(tf.float32, [None, num_visible],
                                    name='vrand')
        self.hrand = tf.placeholder(tf.float32, [None, num_hidden],
                                    name='hrand')
        self.avg_tfe = tf.placeholder(tf.float32, name='avg_tfe')
        self.avg_vfe = tf.placeholder(tf.float32, name='avg_vfe')
        self.reconstruct_loss = tf.placeholder(tf.float32, name='rloss')
        self.lr = tf.placeholder(tf.float32, name='lr')

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
        self.free_energy = self._create_free_energy()
        """
        time_str = strftime("%b-%d-%Y-%H-%M-%S", localtime())
        self.dirname = self.name + '/Run-' + time_str
        self.train_writer = tf.summary.FileWriter(self.dirname)
        self._create_summaries()
        self.merged_summary = tf.summary.merge_all()
        """
        self.sess = tf.Session()
        self.saver = tf.train.Saver(self.weights)
        if restore_dir is not None:
            self.session_path = restore_dir
            self.saver.restore(self.sess, self.session_dir)
            print('Model restored')
        else:
            init = tf.global_variables_initializer()
            self.session_path = self.dirname + '%dby%d' % (num_visible,
                                                           num_hidden)
            self.sess.run(init)
            print('RBM Model built and initialized')

    def _initialize_weights(self):
        weights = dict()
        weights['w'] = tf.Variable(tf.random_normal([self.num_visible,
                                                     self.num_hidden],
                                                    0.0, 0.01),
                                   name='w')
        weights['bh'] = tf.Variable(tf.zeros(shape=[self.num_hidden],
                                             dtype=tf.float32), name='bh')
        weights['bv'] = tf.Variable(tf.zeros(shape=[self.num_visible],
                                             dtype=tf.float32), name='bv')
        return weights

    def _create_goodness(self):
        association = tf.matmul(tf.transpose(self.v), self.encode)
        neg_energy = tf.reduce_mean(tf.multiply(self.weights['w'], association))
        return neg_energy

    def sample_visible_from_hidden(self, hidden):
        logit = tf.matmul(hidden, tf.transpose(self.weights['w']))
        vprob = self.trans_func(tf.add(logit, self.weights['bv']))
        vstate = sample_prob_dist(vprob, self.vrand)
        return vprob, vstate

    def sample_hidden_from_visible(self, visible):
        logit = tf.matmul(visible, self.weights['w'])
        hprob = self.trans_func(tf.add(logit, self.weights['bh']))
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

        """Use noise-free hidden units in reconstruction
        e.g. use prob instead of sampled binary states"""
        pos_association = tf.matmul(tf.transpose(self.v), pos_hprob)
        neg_association = tf.matmul(tf.transpose(neg_vprob), neg_hprob)

        gradient_w = (pos_association - neg_association) / self.batch_size
        update_w = tf.assign_add(self.weights['w'], self.lr * gradient_w)

        g_bh = self.lr * tf.reduce_mean(tf.subtract(pos_hprob, neg_hprob), 0)
        update_bh = tf.assign_add(self.weights['bh'], g_bh)

        g_bv = self.lr * tf.reduce_mean(tf.subtract(self.v, neg_vprob), 0)
        update_bv = tf.assign_add(self.weights['bv'], g_bv)

        # mean squared error
        loss = 0.5 * tf.reduce_sum(tf.square(tf.subtract(neg_vprob, self.v)))

        # return [loss, update_w]
        return [loss, update_w, update_bh, update_bv]

    def _create_reconstruct_loss(self):
        loss = 0.5 * tf.reduce_mean(tf.square(self.reconstruct - self.v))
        return loss

    def _create_free_energy(self):
        first_term = tf.matmul(self.v, tf.reshape(self.weights['bv'],
                                                  [self.num_visible, 1]))
        x = tf.exp(tf.matmul(self.v, self.weights['w']))
        one_plus_x = tf.add(self.one, x)
        second_term = tf.reduce_sum(tf.log(one_plus_x), 1, keep_dims=True)
        return -tf.add(first_term, second_term)

    def _create_summaries(self):
        layer_weight = tf.transpose(self.weights['w'])
        x_min = tf.reduce_min(layer_weight)
        x_max = tf.reduce_max(layer_weight)
        normalized_layer_weight = tf.div(layer_weight - x_min, x_max - x_min)
        """
        instead of show the visible to hidden weights, record the hidden to
        visible units weights, suggested by Hinton's paper
        """
        weight_by_neuron = tf.reshape(normalized_layer_weight,
                                      [self.num_hidden, self.num_visible, 1, 1])
        tf.summary.image('h2v weights', weight_by_neuron, max_outputs=16)
        # tf.summary.image('all 0 is black image',
        # tf.zeros([1, self.num_visible, 1, 1]))

        tf.summary.histogram('histogram of visible to hidden weights',
                             layer_weight)
        tf.summary.histogram('histogram of visible to hidden biases',
                             self.weights['bv'])
        tf.summary.histogram('histogram of hidden to visible biases',
                             self.weights['bh'])
        # tf.summary.scalar('min weight', x_min)
        # tf.summary.scalar('max weight', x_max)
        # mean = tf.reduce_mean(layer_weight)
        # tf.summary.scalar('mean weight', mean)
        # stddev = tf.sqrt(tf.reduce_mean(tf.square(layer_weight - mean)))
        # tf.summary.scalar('stddev weight', stddev)
        """If valid free energy increase relatively to train free energy,
        we are overfitting"""
        tf.summary.scalar('valid FE - train FE', self.avg_vfe - self.avg_tfe)
        tf.summary.scalar('Train Free Energy(TFE)', self.avg_tfe)
        tf.summary.scalar('Valid Free Energy(VFE)', self.avg_vfe)
        tf.summary.scalar('Train reconstruct loss', self.reconstruct_loss)
        tf.summary.scalar('num hidden units', self.num_hidden)
        tf.summary.scalar('learning rate', self.lr)

    def run_train_step(self, V, lr):
        Vrand = np.random.random([V.shape[0], self.num_visible])
        Hrand = np.random.random((V.shape[0], self.num_hidden))
        return self.sess.run(self.updates,
                             feed_dict={self.v: V, self.vrand: Vrand,
                                        self.hrand: Hrand,
                                        self.lr: lr})

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

    def calculate_free_energy(self, V):
        """Use free energy to do classification,
        with combined data vector and label vector"""
        all_ones = np.ones([V.shape[0], self.num_hidden])
        return self.sess.run(self.free_energy, feed_dict={self.v: V,
                                                          self.one: all_ones})

    def train(self, train_dataset, num_steps):
        display_steps = num_steps / 10
        for step in range(num_steps):
            offset = (self.batch_size * step) % (train_dataset.shape[0] -
                                                 self.batch_size)
            end = (offset + self.batch_size) % train_dataset.shape[0]
            if end < offset:
                batch_data = np.concatenate((train_dataset[offset:, :],
                                             train_dataset[:end, :]), axis=0)
            else:
                batch_data = train_dataset[offset:(offset + self.batch_size), :]

            l, w, bh, bv = self.run_train_step(batch_data)
            if step % display_steps == 0:
                batch_loss = self.calc_reconstruct_loss(batch_data)
                print("Batch loss at step %d: %f" % (step, l))
                print("Batch reconstruction loss at step %d: %f" % (step,
                                                                    batch_loss))

        print('Restricted Boltzmann Machine trained')
        train_loss = self.calc_reconstruct_loss(train_dataset)
        print("Trainset reconstruction loss: %f" % train_loss)

    def train_with_labels(self, train_dataset, train_labels, num_steps,
                          valid_dataset, init_lr=0.1):
        display_step = num_steps // 10
        # summary_step = num_steps // 10

        num_labels = train_labels.shape[1]
        print('Training for %d steps' % num_steps)
        y = np.argmax(train_labels, 1)
        X = np.array([train_dataset[y == i, :] for i in range(num_labels)])
        Y = np.array([train_labels[y == i, :] for i in range(num_labels)])

        self.batch_size /= num_labels
        for step in range(num_steps):
            if self.num_labels > 2:
                train0, label0 = get_batch(X[0], Y[0], step, self.batch_size)
                train1, label1 = get_batch(X[1], Y[1], step, self.batch_size)
                train2, label2 = get_batch(X[2], Y[2], step, self.batch_size)
                # train3 = X[3][np.random.choice(X[3].shape[0], 50), :]
                train3, label3 = get_batch(X[3], Y[3], step, self.batch_size)
                train4, label4 = get_batch(X[4], Y[4], step, self.batch_size)
                batch_data = np.concatenate((train0, train1, train2,
                                             train3, train4),
                                            axis=0)
            else:
                train0, label0 = get_batch(X[0], Y[0], step, self.batch_size)
                train1, label1 = get_batch(X[1], Y[1], step, self.batch_size)
                batch_data = np.concatenate((train0, train1), axis=0)

            perm = np.random.permutation(batch_data.shape[0])
            batch_data = batch_data[perm, :]
            lr = init_lr * np.power(0.64, float(step) / float(num_steps))
            l, w, bh, bv = self.run_train_step(batch_data, lr)
            """
            if step % summary_step == 0:
                batch_loss = self.calc_reconstruct_loss(batch_data)
                train_free_energy = self.calculate_free_energy(train_dataset)
                valid_free_energy = self.calculate_free_energy(valid_dataset)
                avg_tfe = np.mean(train_free_energy)
                avg_vfe = np.mean(valid_free_energy)
                summary = self.sess.run(self.merged_summary,
                                        feed_dict={self.avg_tfe: avg_tfe,
                                                   self.avg_vfe: avg_vfe,
                                                   self.reconstruct_loss:
                                                   batch_loss,
                                                   self.lr: lr})
                self.train_writer.add_summary(summary, step)
            """
            if step % display_step == 0:
                batch_loss = self.calc_reconstruct_loss(batch_data)
                train_free_energy = self.calculate_free_energy(train_dataset)
                valid_free_energy = self.calculate_free_energy(valid_dataset)
                avg_tfe = np.mean(train_free_energy)
                avg_vfe = np.mean(valid_free_energy)
                print("Batch loss at step %d: %.6f(lr=%.6f)" % (step, l, lr))
                print("Batch reconstruction loss at step %d: %f" %
                      (step, batch_loss))
                print("Avg Free Energy: Valid(%.4f) - Train(%.4f) = %.4f" %
                      (avg_vfe, avg_tfe, avg_vfe - avg_tfe))

        print('Restricted Boltzmann Machine trained')
        # train_loss = self.calc_reconstruct_loss(train_dataset)
        # print("Trainset reconstruction loss: %f" % train_loss)
        train_free_energy = self.calculate_free_energy(train_dataset)
        valid_free_energy = self.calculate_free_energy(valid_dataset)
        print("Avg Free Energy for Trainset: %f" % np.mean(train_free_energy))
        print("Avg Free Energy for Validset: %f" % np.mean(valid_free_energy))
        # self.save_variables()

    def get_weights(self, name):
        return self.sess.run(self.weights[name])

    def save_variables(self):
        save_path = self.saver.save(self.sess, self.session_path)
        print("Model saved to file", save_path)
