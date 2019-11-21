import numpy as np
import tensorflow as tf
import keras
import os


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

batch_size = 50
num_classes = 10
weight_decay = 1e-4

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
)

def base_model(filters, name):
    c1 = Conv2D(filters, (3, 3), padding='same', name='base1_' + name, kernel_regularizer=regularizers.l2(weight_decay),
                input_shape=x_train.shape[1:])(inputs)
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


def gatingNetwork():
    c1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),
                input_shape=x_train.shape[1:], name='gate1')(inputs)
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

    model = Model(inputs=inputs, outputs=c26)
    return model

''''
pairs:
0 (airplane) 8 (ship)
1 (automobile) 9 (truck)
2 (bird) 6 (frog)
3 (cat) 5 (dog)
4 (deer) 7 (horse)
'''
# labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
def train_base_models(weights_file_in):
    for i in range(5):
        if (i == 0):
            second = 8
        if (i == 1):
            second = 9
        if (i == 2):
            second = 6
        if (i == 3):
            second = 5
        if (i == 4):
            second = 7
        model = models[i]
        weights_file = weights_file_in + str(i)
        checkpointer = ModelCheckpoint(weights_file + '.h5', monitor='val_acc', verbose=1, save_best_only=True,
                                       save_weights_only=True, mode='auto')
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=2, min_lr=0.000001, verbose=1)
        earlystopper = EarlyStopping(monitor='val_acc', min_delta=0.00001, patience=10, verbose=1, mode='auto')
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        callbacks_list = [reduce_lr, earlystopper, checkpointer]

        y = [j for j in range(len(y_train)) if y_train[j][i] == 1 or y_train[j][second] == 1]
        x = x_train[y]
        y = y_train[y]

        y_val = [j for j in range(len(y_test)) if y_test[j][i] == 1 or y_test[j][second] == 1]
        x_val = x_test[y_val]
        y_val = y_test[y_val]

        model.fit_generator(datagen.flow(x, y, batch_size=batch_size),
                            epochs=100,
                            steps_per_epoch=len(x) / batch_size,
                            validation_data=(x_val, y_val), callbacks=callbacks_list,
                            workers=4, verbose=2)

# Loading base model weights
def load_expert_weights_and_set_trainable_layers(model, experts,weights_file='lib/weights/base_model_'):
    for a in range(len(experts)):
        m = experts[a]
        file = weights_file + str(a) + '.h5'
        m.load_weights(file, by_name=True)
        for b in m.layers:
            for l in model.layers:
                if (l.name == b.name):
                    l.set_weights(b.get_weights())
                    print("loaded layer "+str(l.name))

    for l in model.layers:
        if ('gate' in l.name or 'lambda' in l.name):
            l.trainable = True
            # print("training gate ")
        else:
            l.trainable = False

def load_gate_weights(model, model_old,weights_file='lib/weights/moe_full.hdf5'):
    model_old.load_weights(weights_file)
    for l in model.layers:
            for b in model_old.layers:
                if (l.name == b.name):
                    l.set_weights(b.get_weights())
                    print("loaded gate layer "+str(l.name))

def gating_multiplier(gate,branches):
    forLambda=[gate]
    forLambda.extend(branches)
    add = Lambda(lambda x:K.tf.transpose(
        sum(K.tf.transpose(forLambda[i]) *
            forLambda[0][:, i-1] for i in range(1,len(forLambda))
           )
    ))(forLambda)
    return add
# labels=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def train_gate(model, weights_file):
        history = History()
        highest_acc = 0
        iterationsWithoutImprovement = 0
        lr = .001
        for i in range(7):
            # load_weights()
            model.fit_generator(datagen.flow(x_train, y_train, batch_size=50),
                                       epochs=1,
                                       steps_per_epoch=len(x_train) / 50,
                                       validation_data=(x_test, y_test), callbacks=[history],
                                       workers=4, verbose=1)
            val_acc = history.history['val_acc'][-1]
            if (val_acc > highest_acc):
                model.save_weights(weights_file + '.hdf5')
                print("Saving weights, new highest accuracy: " + str(val_acc))
                highest_acc = val_acc
                iterationsWithoutImprovement = 0
            else:
                iterationsWithoutImprovement += 1
                if (iterationsWithoutImprovement > 3):
                    lr *= .5
                    K.set_value(model.optimizer.lr, lr)
                    print("Learning rate reduced to: " + str(lr))
                    iterationsWithoutImprovement = 0

def create_gate_model(expert_models):
    gate_network = gatingNetwork()
    merged = gating_multiplier(gate_network.layers[-1].output, [m.layers[-1].output for m in expert_models])
    b = Activation('softmax', name='gatex')(merged)
    model = Model(inputs=inputs, outputs=b)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

inputs = Input(shape=x_train.shape[1:])

models=[base_model(32,"1"),base_model(32,"2"),base_model(32,"3"),base_model(32,"4"),base_model(32,"5")]

'''
these are pretrained, no need to train again
base_weights_file='weights/base_model_'
train_base=False
if(train_base):
    train_base_models(base_weights_file)
'''

moe_weights_file='lib/weights/moe_full'
for i in range(1,len(models)):

    model=create_gate_model(models[:i])
    if i>1:
        load_gate_weights(model,model_previous)
    load_expert_weights_and_set_trainable_layers(model, models[:i])
    train_gate(model, moe_weights_file)
    model_previous=model