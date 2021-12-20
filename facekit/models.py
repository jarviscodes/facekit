from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LayerNormalization


class GenderClassifierModel(object):
    def __init__(self, show_summary=True):
        # Conv Net
        self.model = Sequential()
        self.model.add(
            Conv2D(input_shape=(224, 224, 3),
                   filters=96,
                   kernel_size=(7, 7),
                   strides=4,
                   padding='valid',
                   activation='relu',
                   name='convolution_1'
                   )
        )

        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name='pooling_1'
            )
        )

        self.model.add(
            LayerNormalization(name='layer_norm_1')
        )

        self.model.add(
            Conv2D(
                filters=256,
                kernel_size=(5, 5),
                strides=1,
                padding='same',
                activation='relu',
                name='convolution_2'
            )
        )

        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                name='pooling_2'
            )
        )

        self.model.add(
            LayerNormalization(name='layer_norm_2')
        )

        self.model.add(
            Conv2D(
                filters=256,
                kernel_size=(3, 3),
                strides=1,
                padding='same',
                activation='relu',
                name='convolution_3'
            )
        )

        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling_3'))
        self.model.add(LayerNormalization(name='layer_norm_3'))

        # FCN
        self.model.add(Flatten(name='flatten_1'))
        self.model.add(Dense(units=512, activation='relu', name='dense_1'))
        self.model.add(Dropout(rate=0.25, name='dropout_1'))
        self.model.add(Dense(units=512, activation='relu', name='dense_2'))
        self.model.add(Dropout(rate=0.25, name='dropout_2'))
        self.model.add(Dense(units=2, activation='softmax', name='dense_3'))
        if show_summary:
            self.model.summary()

