from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from architectures import Architecture

class CascadeNet(Architecture):
    @staticmethod
    def build(width, height, depth, classes, fchead):
        input_shape, channel_dim = Architecture.get_channel_dim(width, height, depth)

        model = Sequential()

        model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape, activation='relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        Architecture.add_fchead(model=model, fchead=fchead, channel_dim=channel_dim, classes=classes)

        return model
