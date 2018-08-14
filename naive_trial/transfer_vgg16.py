import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Input
from keras.models import Model
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.utils import plot_model
from keras.models import model_from_json

#from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from keras.models import load_model
import itertools
import matplotlib.pyplot as plt
from scipy import misc
import pickle
import os
import random as rnd


class event_Callback(keras.callbacks.Callback):
    

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch%20 == 0:
            model_json = model.to_json()
            with open("../models/transfer_vgg16/model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/transfer_vgg16/model.h5")
            print("Saved model to disk")
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        return

def data_argumentation(rootPath, raw_data_files, label, inputSize=(224,224), countLimit=None, random = False):
    
    fCount = 0
    roateAngle = [0, 90, -90, 180]
    procssed_data = []
    procssed_label = []
    
    if random: rnd.shuffle(raw_data_files)
    
    for idx, _ in enumerate(raw_data_files):

        _data = misc.imread(os.path.join(rootPath, raw_data_files[idx]))
        _data = misc.imresize(_data, inputSize)
      
        for _, angle in enumerate(roateAngle):
            
            data_tmp = misc.imrotate(_data, angle)
            procssed_data.append(data_tmp/255.)
            procssed_label.append(label)
            
        if countLimit != None and fCount > countLimit:
            break   
        else:
            fCount += 1
            
    return procssed_data, procssed_label


def load_model(graph_path="../models/transfer_vgg16/model.json",
                weight_path="../models/transfer_vgg16/model.h5"):
    json_file = open(graph_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_path)
    print("Loaded model from disk")
    
    return loaded_model



# vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
#                                             input_tensor=None, input_shape=(96,96,3),
#                                             pooling=None,
#                                             classes=2)

vgg16_model = keras.applications.vgg16.VGG16()


model = Sequential()
# model.add(Convolution2D(
#         input_shape= (96,96,3),
#         filters=64,
#         kernel_size=3,
#         strides=1,
#         #kernel_regularizer=regularizers.l2(regularizers_weight),
#         padding='same',      # Padding method
#     ))

for i in range(0,19):
    #vgg16_model.layers[i].trainable = False
    model.add(vgg16_model.layers[i])

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(2, activation='softmax', name='predictions'))
model.summary()
print(model.layers[5].get_output_at(0),model.layers[5].get_output_at(1))

plot_model(model, to_file='model.png')

ab_dataRoot = '../dataset/ICPR2012/abnormal'
ab_imgPath = os.path.join(ab_dataRoot,'data')
ab_raw_data_files = sorted(os.listdir(ab_imgPath))

n_dataRoot = '../dataset/ICPR2012/normal'
n_imgPath = os.path.join(n_dataRoot,'data')
n_raw_data_files = sorted(os.listdir(n_imgPath))

ab_data, ab_label = data_argumentation(ab_imgPath, ab_raw_data_files, [0,1])
n_data, n_label = data_argumentation(n_imgPath, n_raw_data_files, [1,0], countLimit=len(ab_raw_data_files), random=True)

procssed_data = np.array(ab_data + n_data)
procssed_label = np.array(ab_label + n_label)

procssed_data, procssed_label = shuffle(procssed_data, procssed_label)

event_cb = event_Callback()
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
history = model.fit(
                    x=procssed_data, 
                    y=procssed_label, 
                    validation_split=0.2, 
                    epochs=1000, 
                    batch_size=64, 
                    verbose=2,
                    callbacks=[event_cb]
                    )  
y_hat = model.predict(procssed_data[0:10])
print(np.argmax(y_hat[8], axis=-1))
print(np.argmax(procssed_label[8], axis=-1))