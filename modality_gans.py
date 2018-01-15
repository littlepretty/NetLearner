from __future__ import print_function, division
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam, SGD
# from preprocess import unsw, nslkdd

import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import pickle

dopt = Adam(lr=1e-4)
gopt = SGD(lr=1e-4)
vopt = Adam(lr=1e-4)


class ModGAN():
    def __init__(self):
        self.unified_dim = 640
        self.root = 'ModalityGAN/'

        # unsw.generate_dataset(one_hot_encode=True, root_dir=self.root)
        self.X1 = np.load(self.root + 'UNSW/train_dataset.npy')
        # nslkdd.generate_dataset(True, True, root=self.root)
        self.X2 = np.load(self.root + 'NSLKDD/train_dataset.npy')

        self.discriminator1 = self.build_discriminator('D1')
        self.discriminator2 = self.build_discriminator('D2')
        self.generator1 = self.build_generator(self.X1.shape[1], 'unsw')
        self.generator2 = self.build_generator(self.X2.shape[1], 'nsl')

        z1 = Input(shape=(self.X1.shape[1], ), name='unsw')
        z2 = Input(shape=(self.X2.shape[1], ), name='nsl')
        u = self.generator1(z1)
        v = self.generator2(z2)

        valid1_u = self.discriminator1(u)
        valid2_u = self.discriminator2(u)
        self.combined1 = Model(inputs=z1, outputs=[valid1_u, valid2_u],
                               name='GAN_unsw')
        self.combined1.compile(vopt, 'categorical_crossentropy',
                               metrics=['accuracy'], loss_weights=[.2, 1.])
        self.combined1.summary()

        valid1_v = self.discriminator1(v)
        valid2_v = self.discriminator2(v)
        self.combined2 = Model(inputs=z2, outputs=[valid1_v, valid2_v],
                               name='GAN_nsl')
        self.combined2.compile(vopt, 'categorical_crossentropy',
                               metrics=['accuracy'], loss_weights=[1., .2])
        self.combined2.summary()

    def build_generator(self, feature_dim, input_name):
        hidden = [320, 512, self.unified_dim]
        input_layer = Input(shape=(feature_dim, ), name=input_name)
        H = BatchNormalization()(input_layer)
        H = Dense(hidden[0])(H)
        H = LeakyReLU(alpha=0.2)(H)

        H = BatchNormalization()(H)
        H = Dense(hidden[1])(H)
        H = LeakyReLU(alpha=0.2)(H)

        H = BatchNormalization()(H)
        H = Dense(hidden[2])(H)
        V = Activation('sigmoid')(H)
        generator = Model(input_layer, V, name='G_' + input_name)
        generator.compile(dopt, 'binary_crossentropy')
        # generator.summary()
        return generator

    def build_discriminator(self, model_name):
        hidden = [self.unified_dim, 256, 2]
        drop_prob = 0.2
        input_layer = Input(shape=(self.unified_dim, ))
        H = Dense(hidden[0])(input_layer)
        H = LeakyReLU(alpha=0.2)(H)
        H = Dropout(drop_prob)(H)

        H = Dense(hidden[1])(H)
        H = LeakyReLU(alpha=0.2)(H)
        H = Dropout(drop_prob)(H)

        V = Dense(hidden[2], activation='softmax')(H)
        discriminator = Model(input_layer, V, name=model_name)
        discriminator.compile(dopt, 'categorical_crossentropy',
                              metrics=['accuracy'])
        # discriminator.summary()
        return discriminator

    def train(self, epochs, batch_size=100):
        show_interval = 100
        store_interval = 50
        history = {'d1_loss': [], 'd1_accu': [],
                   'd2_loss': [], 'd2_accu': [],
                   'g1_loss': [], 'g1_accu1': [], 'g1_accu2': [],
                   'g2_loss': [], 'g2_accu1': [], 'g2_accu2': []}
        """Labels for D1"""
        y1 = np.zeros([2 * batch_size, 2])
        y1[0: batch_size, 1] = 1
        y1[batch_size:, 0] = 1
        """Labels for D2"""
        y2 = np.zeros([2 * batch_size, 2])
        y2[0: batch_size, 0] = 1
        y2[batch_size:, 1] = 1
        """Labels for both GANs"""
        y = np.zeros([batch_size, 2])
        y[:, 1] = 1

        def make_trainable(net, val):
            for layer in net.layers:
                layer.trainable = val

        def store_history():
            history['d1_loss'].append(d1_score[0])
            history['d2_loss'].append(d2_score[0])
            history['d1_accu'].append(d1_score[1])
            history['d2_accu'].append(d2_score[1])
            history['g1_loss'].append(g1_score[0])
            history['g2_loss'].append(g2_score[0])
            history['g1_accu1'].append(g1_score[3])
            history['g2_accu1'].append(g2_score[3])
            history['g1_accu2'].append(g1_score[4])
            history['g2_accu2'].append(g2_score[4])

        def plot_hist_accu(idx):
            print('D1 %s = %s' % (self.discriminator1.metrics_names, d1_score))
            print('D2 %s = %s' % (self.discriminator2.metrics_names, d2_score))
            print('G1D1D2 %s = %s' % (self.combined1.metrics_names, g1_score))
            print('G2D1D2 %s = %s' % (self.combined2.metrics_names, g2_score))
            plt.figure(0)
            plt.plot(history['d1_loss'], 'r--', label='d1_loss')
            plt.plot(history['d2_loss'], 'r:', label='d2_loss')
            plt.plot(history['g1_loss'], 'b--', label='g1_loss')
            plt.plot(history['g2_loss'], 'g--', label='g2_loss')
            plt.legend()
            plt.savefig(self.root + 'hist_%d.pdf' % idx, format='pdf')
            plt.clf()

            plt.figure(1)
            plt.plot(history['d1_accu'], 'r--', label='d1_accu')
            plt.plot(history['d2_accu'], 'r:', label='d2_accu')
            plt.plot(history['g1_accu1'], 'b--', label='g1_accu1')
            plt.plot(history['g1_accu2'], 'b:', label='g1_accu2')
            plt.plot(history['g2_accu1'], 'g--', label='g2_accu1')
            plt.plot(history['g2_accu2'], 'g:', label='g2_accu2')
            plt.legend()
            plt.savefig(self.root + 'accu_%d.pdf' % idx, format='pdf')
            plt.clf()

        for i in range(epochs):
            idx1 = np.random.randint(0, self.X1.shape[0], batch_size)
            batch1 = self.X1[idx1]
            idx2 = np.random.randint(0, self.X2.shape[0], batch_size)
            batch2 = self.X2[idx2]

            u = self.generator1.predict(batch1)
            v = self.generator2.predict(batch2)
            X = np.concatenate((u, v))

            d1_score = self.discriminator1.train_on_batch(X, y1)
            d2_score = self.discriminator2.train_on_batch(X, y2)
            # t1_score = self.discriminator1.test_on_batch(X, y1)
            # t2_score = self.discriminator2.test_on_batch(X, y2)

            make_trainable(self.discriminator1, False)
            make_trainable(self.discriminator2, False)
            """Train both G1 and G2 to fool D1/D2"""
            self.combined1.compile(vopt, 'categorical_crossentropy',
                                   metrics=['accuracy'], loss_weights=[.2, 1.])
            self.combined2.compile(vopt, 'categorical_crossentropy',
                                   metrics=['accuracy'], loss_weights=[1., .2])
            g1_score = self.combined1.train_on_batch(batch1, [y, y])
            g2_score = self.combined2.train_on_batch(batch2, [y, y])
            make_trainable(self.discriminator1, True)
            make_trainable(self.discriminator2, True)
            # temp1 = self.discriminator1.test_on_batch(X, y1)
            # temp2 = self.discriminator2.test_on_batch(X, y2)
            """Make sure D1/D2 are not updated when training G1/G2"""
            # print('%s == %s' % (t1_score[0], temp1[0]))
            # print('%s == %s' % (t2_score[0], temp2[0]))

            if i % store_interval == 0:
                store_history()

            if i % show_interval == 0:
                plot_hist_accu(i)

        plot_hist_accu(epochs)
        u = self.generator1.predict(self.X1)
        v = self.generator2.predict(self.X2)
        result = {'u': u, 'v': v, 'history': history}
        filename = self.root + 'U%dRuns%d.pkl' % (self.unified_dim, epochs)
        output = open(filename, 'wb+')
        pickle.dump(result, output)
        output.close()


if __name__ == '__main__':
    modgan = ModGAN()
    modgan.train(1600)
