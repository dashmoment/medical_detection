from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import TensorBoard
import keras

from scipy import misc
import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import argparse
import os


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
def data_argumentation(rootPath, raw_data_files, label, inputSize=(96,96), countLimit=None, random = False):
    
    fCount = 0
    #roateAngle = [0, 90, -90, 180]
    roateAngle = [0]
    procssed_data = []
    procssed_label = []
    
    if random: rnd.shuffle(raw_data_files)
    
    for idx, _ in enumerate(raw_data_files):

        _data = misc.imread(os.path.join(rootPath, raw_data_files[idx]))
        _data = misc.imresize(_data, inputSize)
      
        for _, angle in enumerate(roateAngle):
            
            data_tmp = misc.imrotate(_data, angle)
            procssed_data.append(data_tmp)
            procssed_label.append(label)
            
        if countLimit != None and fCount > countLimit:
            break   
        else:
            fCount += 1
            
    return procssed_data, procssed_label

class plot_Callback(keras.callbacks.Callback):
    
    def __init__(self, model, data, batch_size, model_name):
        self.models = model
        self.data = data
        self.batch_size = batch_size
        self.model_name = model_name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch%20 == 0:
            plot_results(self.models, self.data, self.batch_size, self.model_name)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        return

ab_dataRoot = '../dataset/ICPR2012/abnormal'
ab_imgPath = os.path.join(ab_dataRoot,'data')
ab_raw_data_files = sorted(os.listdir(ab_imgPath))

n_dataRoot = '../dataset/ICPR2012/normal'
n_imgPath = os.path.join(n_dataRoot,'data')
n_raw_data_files = sorted(os.listdir(n_imgPath))

ab_data, ab_label = data_argumentation(ab_imgPath, ab_raw_data_files, 1)
#n_data, n_label = data_argumentation(n_imgPath, n_raw_data_files, 0, countLimit=len(ab_raw_data_files), random=True)

# ICPR dataset
(x_train, y_train) = np.array(ab_data[0:int(len(ab_data)*0.8)]), np.array(ab_label[0:int(len(ab_data)*0.8)])
(x_test, y_test) = np.array(ab_data[int(len(ab_data)*0.8):]), np.array(ab_label[int(len(ab_data)*0.8):])

x_train = x_train[:,:,:,0]
x_test = x_test[:,:,:,0]
print(x_train.shape)

image_size = x_train.shape[1]
x_train = np.reshape(x_train, [-1,  x_train.shape[1],  x_train.shape[2],  1])
x_test = np.reshape(x_test, [-1, x_train.shape[1],  x_train.shape[2],  1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print(x_train.shape)

# MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# image_size = x_train.shape[1]
# x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
# x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (image_size, image_size, 1)
batch_size = 128
kernel_size = 3
filters = 128
latent_dim = 128
epochs = 30

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(2):
    filters *= 2
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(16, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for i in range(2):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=2,
                        padding='same')(x)
    filters //= 2

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))

    #reconstruction_loss *= image_size * image_size
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.mean(K.sum(kl_loss, axis=-1))
    kl_loss *= -0.5
    print(kl_loss.shape, reconstruction_loss.shape)
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    if args.weights:
        vae = vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None))
        vae.save_weights('vae_cnn_mnist.h5')

plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")