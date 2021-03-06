from scipy import misc
import numpy as np
import pickle
import os
import random as rnd
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras import regularizers
import keras
from sklearn.utils import shuffle

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
            with open("../models/baseline/model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("../models/baseline/model.h5")
            print("Saved model to disk")
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        #self.losses.append(logs.get('loss'))
        return

def load_model(graph_path="../models/baseline/model.json",
                weight_path="../models/baseline/model.h5"):
    json_file = open(graph_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weight_path)
    print("Loaded model from disk")
    
    return loaded_model


def build_model_fc(input_shape = (16, 16, 1)):
    
    regularizers_weight = 0.001
    
    model = Sequential()
    model.add(Convolution2D(
        input_shape= input_shape,
        filters=32,
        kernel_size=3,
        strides=1,
        kernel_regularizer=regularizers.l2(regularizers_weight),
        padding='same',      # Padding method
    ))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        filters=32,
        kernel_size=3,
        strides=1,
        kernel_regularizer=regularizers.l2(regularizers_weight),
        padding='same',      # Padding method
    ))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D())
    
    model.add(Convolution2D(
        filters=32,
        kernel_size=3,
        strides=1,
        kernel_regularizer=regularizers.l2(regularizers_weight),
        padding='same',      # Padding method
    ))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D())
    
    model.add(Convolution2D(
        filters=32,
        kernel_size=3,
        strides=1,
        kernel_regularizer=regularizers.l2(regularizers_weight),
        padding='same',      # Padding method
    ))
    
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(2048, activation = 'relu'))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    return model

def data_argumentation(rootPath, raw_data_files, label, inputSize=(96,96), countLimit=None, random = False):
    
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

ab_dataRoot = '../dataset/ICPR2012/abnormal'
ab_imgPath = os.path.join(ab_dataRoot,'data')
ab_raw_data_files = sorted(os.listdir(ab_imgPath))

n_dataRoot = '../dataset/ICPR2012/normal'
n_imgPath = os.path.join(n_dataRoot,'data')
n_raw_data_files = sorted(os.listdir(n_imgPath))

ab_data, ab_label = data_argumentation(ab_imgPath, ab_raw_data_files, [0,1])
n_data, n_label = data_argumentation(n_imgPath, n_raw_data_files, [1,0], countLimit=len(ab_raw_data_files), random=True)


# def cutimized_loss(y_true, y_pred):
#      loss_bg = keras.losses.mean_squared_error(y_true[:,:,:,0], y_pred[:,:,:,0])
#      loss_gt = keras.losses.mean_squared_error(y_true[:,:,:,1], y_pred[:,:,:,1])
#      loss = 0.001*loss_bg + loss_gt
     
#      return loss
procssed_data = np.array(ab_data + n_data)
procssed_label = np.array(ab_label + n_label)
procssed_data, procssed_label = shuffle(procssed_data, procssed_label)

model = build_model_fc((96,96,3))
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
