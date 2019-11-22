import numpy as np
import tensorflow as tf
import keras
import os

from model_frame import ModelFrame

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, MaxPooling2D, Conv2D, Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.layers import BatchNormalization, Input
from keras.layers import Concatenate
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import multiply, add
from keras import regularizers
from keras.callbacks import History
from keras.optimizers import Adam

#Training args
batch_size = 50
num_classes = 10
weight_decay = 1e-4

class Expert(ModelFrame):
    def __init__(self, x_train, y_train, x_test, y_test, filters, name, inputs):
        ModelFrame.__init__(self, x_train, y_train, x_test, y_test)
        self.expertModel = self.base_model(filters, name, inputs)

    def base_model(self, filters, name, inputs):
        c1 = Conv2D(filters, (3, 3), padding='same', name='base1_' + name, kernel_regularizer=regularizers.l2(weight_decay),
                    input_shape=self.x_train.shape[1:])(inputs)
        c2 = Activation('elu', name='base2_' + name)(c1)
        c3 = BatchNormalization(name='base3_' + name)(c2)
        c4 = Conv2D(filters, (3, 3), name='base4_' + name, padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay))(c3)
        c5 = Activation('elu', name='base5_' + name)(c4)
        c6 = BatchNormalization(name='base6_' + name)(c5)
        c7 = MaxPooling2D(pool_size=(2, 2), name='base7_' + name)(c6)
        c8 = Dropout(0.2, name='base8_' + name)(c7)

        c9 = Conv2D(filters * 2, (3, 3), name='base9_' + name, padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay))(c8)
        c10 = Activation('elu', name='base10_' + name)(c9)
        c11 = BatchNormalization(name='base11_' + name)(c10)
        c12 = Conv2D(filters * 2, (3, 3), name='base12_' + name, padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay))(c11)
        c13 = Activation('elu', name='base13_' + name)(c12)
        c14 = BatchNormalization(name='base14_' + name)(c13)
        c15 = MaxPooling2D(pool_size=(2, 2), name='base15_' + name)(c14)
        c16 = Dropout(0.3, name='base16_' + name)(c15)

        c17 = Conv2D(filters * 4, (3, 3), name='base17_' + name, padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay))(c16)
        c18 = Activation('elu', name='base18_' + name)(c17)
        c19 = BatchNormalization(name='base19_' + name)(c18)
        c20 = Conv2D(filters * 4, (3, 3), name='base20_' + name, padding='same',
                    kernel_regularizer=regularizers.l2(weight_decay))(c19)
        c21 = Activation('elu', name='base21_' + name)(c20)
        c22 = BatchNormalization(name='base22_' + name)(c21)
        c23 = MaxPooling2D(pool_size=(2, 2), name='base23_' + name)(c22)
        c24 = Dropout(0.4, name='base24_' + name)(c23)

        c25 = Flatten(name='base25_' + name)(c24)
        c26 = Dense(num_classes, name='base26_' + name)(c25)
        c27 = Activation('softmax', name='base27_' + name)(c26)
        return Model(inputs=inputs, outputs=c27)
        