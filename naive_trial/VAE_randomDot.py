'''Example of VAE on MNIST dataset using CNN
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

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
#plt.switch_backend('agg')

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


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector
    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """
    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)
    
    for i in range(5):
        filename = os.path.join('../',model_name, 'decode_result_' + str(i) +'.png')
        filename_input = os.path.join('../',model_name, 'decode_input_' + str(i) +'.png')
        test_img = np.reshape(x_test[i], [-1,  x_test[i].shape[0],  x_test[i].shape[1], x_test[i].shape[2]])
        x_decoded = decoder.predict(encoder.predict(test_img)[2])
        print(x_decoded.shape)
        x_decoded = np.reshape(x_decoded, [-1,  x_decoded.shape[1],  x_decoded.shape[2]])  
        test_img = np.reshape(y_test[i], [-1,  x_test[i].shape[0],  x_test[i].shape[1]])
        plt.imsave(filename_input, test_img[0])
        plt.imsave(filename, x_decoded[0])
       
        print(filename_input, filename)
        print(test_img[0].shape, x_decoded[0].shape)
       

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

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=32, image_height=32, image_width=32):
        'Initialization'
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(200)

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        X, y = self.__data_generation()
        return X, y

    def __data_generation(self):
        
        max_cell_num = 300
        output_batch_image = []
        
        for _ in range(self.batch_size):

            #curr_cell_num = rnd.randint(0, max_cell_num)
            curr_cell_num = 3
            curr_image = np.zeros([self.image_height, self.image_width, 1])
    
            for _ in range(curr_cell_num):
                h_coordinate = rnd.randint(0, self.image_height-1)
                w_coordinate = rnd.randint(0, self.image_width-1)
                
                curr_image[h_coordinate, w_coordinate] = 1.0

            output_batch_image.append(curr_image)

        return  np.array(output_batch_image),  np.array(output_batch_image)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# image_size = x_train.shape[1]
# x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
# x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# ab_dataRoot = '../dataset/ICPR2012/abnormal'
# ab_imgPath = os.path.join(ab_dataRoot,'data')
# ab_labelPath = os.path.join(ab_dataRoot,'label')
# ab_raw_data_files = sorted(os.listdir(ab_imgPath))
# ab_label_files = sorted(os.listdir(ab_labelPath))

# ab_data, ab_label = data_argumentation(ab_imgPath, ab_raw_data_files, ab_labelPath, ab_label_files)


# # ICPR dataset
# (x_train, y_train) = np.array(ab_data[0:int(len(ab_data)*0.8)]), np.array(ab_label[0:int(len(ab_data)*0.8)])
# (x_test, y_test) = np.array(ab_data[int(len(ab_data)*0.8):]), np.array(ab_label[int(len(ab_data)*0.8):])


# image_size = x_train.shape[1]
# x_train = np.reshape(x_train, [-1,  x_train.shape[1],  x_train.shape[2],  x_train.shape[3]])
# x_test = np.reshape(x_test, [-1, x_train.shape[1],  x_train.shape[2],  x_train.shape[3]])
# y_train = np.reshape(y_train, [-1,  y_train.shape[1],  y_train.shape[2],  1])
# y_test = np.reshape(y_test, [-1, y_test.shape[1],  y_test.shape[2], 1])

# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# y_train = y_train.astype('float32') / 255
# y_test = y_test.astype('float32') / 255

# network parameters

#input_shape = ( x_train.shape[1],  x_train.shape[2],  x_train.shape[3])
input_shape = (28, 28, 1)
batch_size = 64
kernel_size = 3
filters = 64
latent_dim = 128
epochs = 1000

params = {'batch_size': batch_size,
          'image_height': input_shape[0],
          'image_width': input_shape[1],
          }

training_generator = DataGenerator(**params)

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
x = Dense(1024, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
#plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

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
#plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

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
    
    #Generate validation Set
    x_test = []
    y_test = []
    for i in range(10):
        _tmp =  training_generator.__getitem__(i)
        x_test.append(_tmp[0])
        y_test.append(_tmp[1])
    
    x_test = np.vstack(x_test)  
    y_test = np.vstack(y_test)  
    
    data = (x_test, x_test)
    print(x_test.shape)
    

    def loss_function(z_log_var, z_mean):
        
        def vae_loss(ytrue, ypred):
            
            #reconstruction_loss = binary_crossentropy(K.flatten(ytrue), K.flatten(ypred))*32*32
            reconstruction_loss = K.sqrt(K.sum(K.square(ytrue - ypred), axis = [1,2,3]))
            #reconstruction_loss = K.sum(K.abs(ytrue - ypred), axis = [1,2,3])
            #reconstruction_loss = -K.mean(inputs * K.log(1e-10 + outputs)
            #                                        + (1-inputs) * K.log(1e-10 + 1 - outputs), [1,2,3])    
            kl_loss = - 0.5 * K.mean(K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
            vae_loss = K.mean(reconstruction_loss + kl_loss)

            return vae_loss
        return vae_loss
   
    #vae.add_loss(vae_loss)
    vae.compile(
                optimizer='rmsprop', 
                loss=loss_function(z_log_var, z_mean)
                )
    vae.summary()

    plot_cb = plot_Callback(models, data, batch_size, "vae_cnn")
    tensorboard = TensorBoard(log_dir='../logs', histogram_freq=0,
                          write_graph=True, write_images=True)

    if args.weights:
        vae = vae.load_weights(args.weights)
    else:
        # train the autoencoder
        
        vae.fit_generator(
                            generator=training_generator,
                            epochs=epochs,
                            validation_data=data,
                            callbacks=[tensorboard, plot_cb],
                            use_multiprocessing=True,
                            workers=6
                        )
        '''
        vae.fit(x_train,
                x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, x_test),
                callbacks=[tensorboard, plot_cb]
                )
        '''
        vae.save_weights('../vae_cnn_ICPR.h5')
        print('save weight')
