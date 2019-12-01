import numpy as np
import tensorflow as tf
import keras
import os

from model_frame import ModelFrame

from elephas.spark_model import SparkModel

from tensorflow.keraas.datasets import cifar10
from tensorflow.keraas.preprocessing.image import ImageDataGenerator
from tensorflow.keraas.models import Sequential
from tensorflow.keraas.layers import Input, Dense, Dropout, Activation, Flatten, MaxPooling2D, Conv2D, Reshape, Conv2DTranspose
from tensorflow.keraas.models import Model
from tensorflow.keraas import backend as K
from tensorflow.keraas import models
from tensorflow.keraas.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keraas.layers import BatchNormalization, Input
from tensorflow.keraas.layers import Concatenate
from tensorflow.keraas.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from tensorflow.keraas.layers import multiply, add
from tensorflow.keraas import regularizers
from tensorflow.keraas.callbacks import History
from tensorflow.keraas.optimizers import Adam

#Training args
batch_size = 50
num_classes = 10
weight_decay = 1e-4

class SparseGate(ModelFrame):
    def __init__(self, x_train, y_train, x_test, y_test, inputs, spark_context):
        ModelFrame.__init__(self, x_train, y_train, x_test, y_test, spark_context)
        self.gateModel = None
        self.inputs = inputs
        
    def gating_network(self):
        c1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                    input_shape=self.x_train.shape[1:], name='gate1')(self.inputs)
        c2 = Activation('elu', name='gate2')(c1)
        c3 = BatchNormalization(name='gate3')(c2)
        c4 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='gate4')(c3)
        c5 = Activation('elu', name='gate5')(c4)
        c6 = BatchNormalization(name='gate6')(c5)
        c7 = MaxPooling2D(pool_size=(2, 2), name='gate7')(c6)
        c8 = Dropout(0.2, name='gate26')(c7)
        c9 = Conv2D(32 * 2, (3, 3), name='gate8', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(c8)
        c10 = Activation('elu', name='gate9')(c9)
        c11 = BatchNormalization(name='gate25')(c10)
        c12 = Conv2D(32 * 2, (3, 3), name='gate10', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(c11)
        c13 = Activation('elu', name='gate11')(c12)
        c14 = BatchNormalization(name='gate12')(c13)
        c15 = MaxPooling2D(pool_size=(2, 2), name='gate13')(c14)
        c16 = Dropout(0.3, name='gate14')(c15)

        c25 = Flatten(name='gate23')(c16)
        c26 = Dense(5, name='gate24', activation='elu')(c25)

        model = Model(inputs=self.inputs, outputs=c26)
        return model

    def create_gate_model(self,expert_models):
        gate_network = self.gating_network()
        merged = merged =Lambda(lambda x:K.tf.transpose(
            sum(K.tf.transpose(x[i]) *
                x[0][:, i-1] for i in range(1,len(x))
            )
        ))([gate_network.layers[-1].output]+[m.layers[-1].output for m in expert_models])
        b = Activation('softmax', name='gatex')(merged)
        model = Model(inputs=self.inputs, outputs=b)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model

    def train_gate(self, datagen, weights_file):
        model = self.gateModel
        self.gateModel = SparkModel(model, frequency='epoch', mode='asynchronous')
        self.gateModel.fit(self.rdd, epochs=1, batch_size=50, verbose=1, 
                                validation_split = 0.1)
        self.gateModel = self.gateModel.master_network
        self.gateModel.save_weights(weights_file + '.hdf5')

    def load_gate_weights(self, model_old,weights_file='../lib/weights/moe_full.hdf5'):
        model_old.load_weights(weights_file)
        for l in self.gateModel.layers:
                for b in model_old.layers:
                    if (l.name == b.name):
                        l.set_weights(b.get_weights())
                        print("loaded gate layer "+str(l.name))