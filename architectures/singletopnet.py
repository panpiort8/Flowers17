from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from architectures import Architecture
from keras import backend as K

class SingleTopNet:
    @staticmethod
    def build(input_shape, layers, classes):
        channel_dim = -1
        if K.image_data_format() == 'channels_first':
            channel_dim = 1

        model = Sequential()
        model.add(Flatten(input_shape=input_shape))

        for layer in layers:
            model.add(Dense(layer, activation='relu'))
            model.add(BatchNormalization(axis=channel_dim))
            model.add(Dropout(0.5))

        model.add(Dense(classes, activation='softmax'))

        return model