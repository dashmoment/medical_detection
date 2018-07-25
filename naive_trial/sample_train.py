import pickle
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential


def build_model(input_shape = (16, 16, 1)):
    model = Sequential()
    model.add(Convolution2D(
        input_shape= input_shape,
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',      # Padding method
    ))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',      # Padding method
    ))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding='same',      # Padding method
    ))
    model.add(Activation('relu'))
    model.add(Convolution2D(
        filters=2,
        kernel_size=3,
        strides=1,
        padding='same',      # Padding method
    ))
    model.add(Activation('softmax'))

    print(model.summary())

    return model


data = pickle.load(open('samples/wh16_s5000/data.pkl','rb'))
label = pickle.load(open('samples/wh16_s5000/label.pkl','rb'))
# print(type(label), label.shape)
# plt.imshow(label[11,:,:,0])
# plt.show()

model = build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
model.fit(x=data, y=label, validation_split=0.2, epochs=100, batch_size=64, verbose=2)  

y_hat = model.predict(data[0:10])

plt.imshow(y_hat[0,:,:,0])
plt.show()
plt.imshow(y_hat[0,:,:,1])
plt.show()

plt.imshow(label[0,:,:,0])
plt.show()