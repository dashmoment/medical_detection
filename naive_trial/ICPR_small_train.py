from scipy import misc
import numpy as np
import pickle
import os
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras import regularizers
import keras


def build_model(input_shape = (16, 16, 1)):
    
    regularizers_weight = 0.0001
    
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
    model.add(MaxPooling2D())
    model.add(Convolution2D(
        filters=2,
        kernel_size=3,
        strides=1,
        #kernel_regularizer=regularizers.l2(regularizers_weight),
        padding='same',      # Padding method
    ))
    model.add(Activation('softmax'))

    print(model.summary())

    return model


dataRoot = '../dataset/ICPR2012/abnormal'
imgPath = os.path.join(dataRoot,'data')
labelPath = os.path.join(dataRoot,'label')

raw_data_files = sorted(os.listdir(imgPath))
raw_label_files = sorted(os.listdir(labelPath)) 

#Do data argumentation
procssed_data = []
procssed_label = []

roateAngle = [0, 90, -90, 180]
fid = 0
for idx, _ in enumerate(raw_data_files):
    
    
    name_prefix = raw_data_files[idx].split('.')[0]

    assert raw_data_files[idx] == raw_label_files[idx], "Data and label is not match"
    _data = misc.imread(os.path.join(imgPath, raw_data_files[idx]))
    _label = misc.imread(os.path.join(labelPath, raw_label_files[idx]))
    _data = misc.imresize(_data, (96,96))
    _label =  misc.imresize(_label, (12,12))
  
    for _, angle in enumerate(roateAngle):
        
        data_tmp = misc.imrotate(_data, angle)
        procssed_data.append(data_tmp/255.)
        
        label_tmp = misc.imrotate(_label, angle)
        layer_bg = label_tmp == 0
        layer_label = label_tmp != 0
        layer_bg = layer_bg.astype(np.float)
        layer_label = layer_label.astype(np.float)
        procssed_label.append(np.dstack([layer_bg,layer_label]))
        
#        fid+=1
#    if fid>100: break
        
        
def cutimized_loss(y_true, y_pred):
     loss_bg = keras.losses.mean_squared_error(y_true[:,:,:,0], y_pred[:,:,:,0])
     loss_gt = keras.losses.mean_squared_error(y_true[:,:,:,1], y_pred[:,:,:,1])
     loss = 0.001*loss_bg + loss_gt
     
     return loss

procssed_data = np.array(procssed_data)
procssed_label = np.array(procssed_label)
print(procssed_data.shape)
print(procssed_label.shape)
model = build_model((96,96,3))
model.compile(loss=cutimized_loss, optimizer='adam', metrics=['accuracy']) 
history = model.fit(x=procssed_data, y=procssed_label, validation_split=0.2, epochs=1000, batch_size=64, verbose=2)  
y_hat = model.predict(procssed_data[0:10])
print(np.argmax(y_hat[8], axis=-1))
print(np.argmax(procssed_label[8], axis=-1))
