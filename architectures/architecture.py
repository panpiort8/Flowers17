from keras import backend as K
from keras.layers import Flatten, Dense, BatchNormalization, Dropout


class Architecture:
    @staticmethod
    def build(width, height, depth, classes, fchead):
        pass

    @staticmethod
    def get_channel_dim(width, height, depth):
        if K.image_data_format() == 'channels_first':
            return (depth, height, width), 1
        else:
            return (height, width, depth), -1

    @staticmethod
    def add_fchead(model, fchead, classes, channel_dim):
        model.add(Flatten())
        for layer in fchead:
            model.add(Dense(layer, activation='relu'))
            model.add(BatchNormalization(axis=channel_dim))
            model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))