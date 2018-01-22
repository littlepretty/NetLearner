from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
# from preprocess import unsw, nslkdd
from keras.layers.advanced_activations import LeakyReLU
import pickle
import numpy as np

gan_dir = 'ModalityGAN/'
generator1 = load_model(gan_dir + 'generator1.h5')
generator2 = load_model(gan_dir + 'generator2.h5')

# unsw.generate_dataset(one_hot_encode=True, root_dir=gan_dir)
X1 = np.load(gan_dir + 'UNSW/train_dataset.npy')
y1 = np.load(gan_dir + 'UNSW/train_labels.npy')
tX1 = np.load(gan_dir + 'UNSW/test_dataset.npy')
ty1 = np.load(gan_dir + 'UNSW/test_labels.npy')

# nslkdd.generate_dataset(True, True, root=gan_dir)
X2 = np.load(gan_dir + 'NSLKDD/train_dataset.npy')
y2 = np.load(gan_dir + 'NSLKDD/train_labels.npy')
tX2 = np.load(gan_dir + 'NSLKDD/test_dataset.npy')
ty2 = np.load(gan_dir + 'NSLKDD/test_labels.npy')

u = generator1.predict(X1)
v = generator2.predict(X2)
tu = generator1.predict(tX1)
tv = generator2.predict(tX2)

U = np.concatenate((u, v), axis=0)
y = np.concatenate((y1, y2), axis=0)
tU = np.concatenate((tu, tv), axis=0)
ty = np.concatenate((ty1, ty2), axis=0)

input_layer = Input(shape=(U.shape[1], ))
hidden = [512, 256, 2]
drop_prob = 0.2
opt = Adam(lr=4e-4)
batch_size = 40
epochs = 4

H = BatchNormalization()(input_layer)
H = Dense(hidden[0])(H)
H = LeakyReLU(alpha=0.2)(H)
H = Dropout(drop_prob)(H)

H = BatchNormalization()(H)
H = Dense(hidden[1], activation='sigmoid')(H)
H = Dropout(drop_prob)(H)

output = Dense(hidden[2], activation='softmax')(H)
classifier = Model(input_layer, output, name='Cls')
classifier.compile(opt, 'categorical_crossentropy', metrics=['accuracy'])

hist = classifier.fit(U, y, batch_size, epochs, verbose=1)
result = classifier.evaluate(tU, ty, batch_size, 1, verbose=0)
for (i, name) in enumerate(classifier.metrics_names):
    print('%s = %s' % (name, result[i]))
