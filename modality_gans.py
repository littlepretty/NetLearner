from __future__ import print_function, division
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD
# from preprocess import unsw, nslkdd
from memory_profiler import profile
import matplotlib.pyplot as plt
# import tensorflow as tf
import numpy as np
import pickle


class ModGAN():
    def __init__(self, resume=False, hist_filename=None):
        self.root = 'ModalityGAN/'
        self.unified_dim = 320
        # unsw.generate_dataset(one_hot_encode=True, root_dir=self.root)
        self.X1 = np.load(self.root + 'UNSW/train_dataset.npy')
        # nslkdd.generate_dataset(True, True, root=self.root)
        self.X2 = np.load(self.root + 'NSLKDD/train_dataset.npy')

        if resume is False:
            self.build_all_models()
            self.hist = {'d1_loss': [], 'd1_accu': [],
                         'd2_loss': [], 'd2_accu': [], 'steps': [],
                         'g1_loss': [], 'g1_accu1': [], 'g1_accu2': [],
                         'g2_loss': [], 'g2_accu1': [], 'g2_accu2': []}
        else:
            self.load_models()
            self.combined1.summary()
            self.combined2.summary()
            self.hist = self.load_history(hist_filename)

    def build_all_models(self):
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
                               metrics=['accuracy'], loss_weights=[.5, .5])
        self.combined1.summary()

        valid1_v = self.discriminator1(v)
        valid2_v = self.discriminator2(v)
        self.combined2 = Model(inputs=z2, outputs=[valid1_v, valid2_v],
                               name='GAN_nsl')
        self.combined2.compile(vopt, 'categorical_crossentropy',
                               metrics=['accuracy'], loss_weights=[.5, .5])
        self.combined2.summary()

    def build_generator(self, feature_dim, input_name):
        hidden = [640, 480, self.unified_dim]
        input_layer = Input(shape=(feature_dim, ), name=input_name)
        H = BatchNormalization()(input_layer)
        H = Dense(hidden[0], activation='sigmoid')(H)
        # H = LeakyReLU(alpha=0.2)(H)

        H = BatchNormalization()(H)
        H = Dense(hidden[1], activation='sigmoid')(H)
        # H = LeakyReLU(alpha=0.2)(H)

        H = BatchNormalization()(H)
        V = Dense(hidden[2], activation='sigmoid')(H)
        generator = Model(input_layer, V, name='G_' + input_name)
        generator.compile(dopt, 'binary_crossentropy')
        generator.summary()
        return generator

    def build_discriminator(self, model_name):
        hidden = [256, 128, 2]
        drop_prob = 0.2
        input_layer = Input(shape=(self.unified_dim, ))
        H = BatchNormalization()(input_layer)
        H = Dense(hidden[0], activation='relu')(H)
        # H = LeakyReLU(alpha=0.2)(H)
        H = Dropout(drop_prob)(H)

        H = BatchNormalization()(H)
        H = Dense(hidden[1], activation='sigmoid')(H)
        # H = LeakyReLU(alpha=0.2)(H)
        H = Dropout(drop_prob)(H)

        V = Dense(hidden[2], activation='softmax')(H)
        discriminator = Model(input_layer, V, name=model_name)
        discriminator.compile(dopt, 'categorical_crossentropy',
                              metrics=['accuracy'])
        # discriminator.summary()
        return discriminator

    def make_trainable(self, net, val):
        for layer in net.layers:
            layer.trainable = val

    @profile
    def train(self, num_steps, batch_size=100):
        show_interval = max(1, num_steps / 10)
        store_interval = max(1, num_steps // 100)
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

        init_step = 0
        if len(self.hist['steps']) > 0:
            init_step = self.hist['steps'][-1] + 1

        for i in xrange(init_step, init_step + num_steps + 1):
            idx1 = np.random.randint(0, self.X1.shape[0], batch_size)
            batch1 = self.X1[idx1]
            idx2 = np.random.randint(0, self.X2.shape[0], batch_size)
            batch2 = self.X2[idx2]

            u = self.generator1.predict(batch1, batch_size)
            v = self.generator2.predict(batch2, batch_size)
            X = np.concatenate((u, v), axis=0)

            d1_score = self.discriminator1.train_on_batch(X, y1)
            d2_score = self.discriminator2.train_on_batch(X, y2)
            # t1_score = self.discriminator1.test_on_batch(X, y1)
            # t2_score = self.discriminator2.test_on_batch(X, y2)

            self.make_trainable(self.discriminator1, False)
            self.make_trainable(self.discriminator2, False)
            """Train both G1 and G2 to fool D1/D2"""
            self.combined1.compile(vopt, 'categorical_crossentropy',
                                   metrics=['accuracy'], loss_weights=[.5, .5])
            self.combined2.compile(vopt, 'categorical_crossentropy',
                                   metrics=['accuracy'], loss_weights=[.5, .5])
            g1_score = self.combined1.train_on_batch(batch1, [y, y])
            g2_score = self.combined2.train_on_batch(batch2, [y, y])
            self.make_trainable(self.discriminator1, True)
            self.make_trainable(self.discriminator2, True)
            """Make sure D1/D2 are not updated when training G1/G2
            temp1 = self.discriminator1.test_on_batch(X, y1)
            temp2 = self.discriminator2.test_on_batch(X, y2)
            print('%s == %s' % (t1_score[0], temp1[0]))
            print('%s == %s' % (t2_score[0], temp2[0]))
            """
            if i % store_interval == 0:
                self.store_history(i, d1_score, d2_score, g1_score, g2_score)

            if i % show_interval == 0:
                self.plot_hist_accu(i, d1_score, d2_score, g1_score, g2_score)

        self.checkpoint()
        self.dump_history(init_step + num_steps)

    def store_history(self, idx, d1_score, d2_score, g1_score, g2_score):
        self.hist['steps'].append(idx)
        self.hist['d1_loss'].append(d1_score[0])
        self.hist['d2_loss'].append(d2_score[0])
        self.hist['d1_accu'].append(d1_score[1])
        self.hist['d2_accu'].append(d2_score[1])
        self.hist['g1_loss'].append(g1_score[0])
        self.hist['g2_loss'].append(g2_score[0])
        self.hist['g1_accu1'].append(g1_score[3])
        self.hist['g2_accu1'].append(g2_score[3])
        self.hist['g1_accu2'].append(g1_score[4])
        self.hist['g2_accu2'].append(g2_score[4])

    def plot_hist_accu(self, idx, d1_score, d2_score, g1_score, g2_score):
        print('D1 %s = %s' % (self.discriminator1.metrics_names, d1_score))
        print('D2 %s = %s' % (self.discriminator2.metrics_names, d2_score))
        print('G1D1D2 %s = %s' % (self.combined1.metrics_names, g1_score))
        print('G2D1D2 %s = %s' % (self.combined2.metrics_names, g2_score))
        steps = self.hist['steps']
        plt.figure(0)
        plt.plot(steps, self.hist['d1_loss'], 'r--', label='d1_loss')
        plt.plot(steps, self.hist['d2_loss'], 'm:', label='d2_loss')
        plt.plot(steps, self.hist['g1_loss'], 'b--', label='g1_loss')
        plt.plot(steps, self.hist['g2_loss'], 'g:', label='g2_loss')
        plt.grid(linestyle=':')
        plt.legend()
        plt.savefig(self.root + 'loss_%d.pdf' % idx, format='pdf')
        plt.close()

        plt.figure(1)
        plt.plot(steps, self.hist['d1_accu'], 'r--', label='d1_accu')
        plt.plot(steps, self.hist['d2_accu'], 'm:', label='d2_accu')
        plt.plot(steps, self.hist['g1_accu1'], 'b--', label='g1_accu1')
        plt.plot(steps, self.hist['g1_accu2'], 'b:', label='g1_accu2')
        plt.plot(steps, self.hist['g2_accu1'], 'g--', label='g2_accu1')
        plt.plot(steps, self.hist['g2_accu2'], 'g:', label='g2_accu2')
        plt.grid(linestyle=':')
        plt.legend()
        plt.savefig(self.root + 'accu_%d.pdf' % idx, format='pdf')
        plt.close()

    def checkpoint(self):
        self.generator1.save(self.root + 'generator1.h5')
        self.generator2.save(self.root + 'generator2.h5')
        self.discriminator1.save(self.root + 'discriminator1.h5')
        self.discriminator2.save(self.root + 'discriminator2.h5')
        self.combined1.save(self.root + 'combined1.h5')
        self.combined2.save(self.root + 'combined2.h5')

    def load_models(self):
        self.generator1 = load_model(self.root + 'generator1.h5')
        self.generator2 = load_model(self.root + 'generator2.h5')
        self.discriminator1 = load_model(self.root + 'discriminator1.h5')
        self.discriminator2 = load_model(self.root + 'discriminator2.h5')
        self.combined1 = load_model(self.root + 'combined1.h5')
        self.combined2 = load_model(self.root + 'combined2.h5')

    def load_history(self, hist_filename):
        f = open(self.root + hist_filename, 'rb')
        hist = pickle.load(f)
        f.close()
        return hist

    def dump_history(self, steps):
        filename = self.root + 'U%dRuns%d.pkl' % (self.unified_dim, steps)
        output = open(filename, 'wb')
        pickle.dump(self.hist, output)
        output.close()


if __name__ == '__main__':
    dopt = Adam(lr=1e-4)
    gopt = SGD(lr=4e-4)
    vopt = Adam(lr=1e-4)
    n1, n2 = 600, 400
    modgan = ModGAN()
    modgan.train(n1)
    # pkl_file = 'U320Runs%d.pkl' % n1
    # modgan = ModGAN(True, pkl_file)
    # modgan.train(n2)
