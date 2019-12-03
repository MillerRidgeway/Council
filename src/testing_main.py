from expert import Expert
from mixture import Mixture

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.layers import Input

from pyspark import SparkContext, SparkConf
import os

os.environ["CUDA_VISIBLE_DEVICES"]=""

num_classes = 10

#Data Loading
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

conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[2]').set(f"spark.executorEnv.CUDA_VISIBLE_DEVICES",' ')
sc = SparkContext(conf=conf)
print(sc._conf.getAll())
#sc=None

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

inputs = Input(shape=x_train.shape[1:])

#Load init experts
experts = []
for i in range(5):
    tempExpert = Expert(x_train,y_train,x_test,y_test, 32, str(i + 1), inputs)
    experts.append(tempExpert.expertModel)

#Storage dir for MoE weights
moe_weights_file='../lib/weights/moe_full'

# Convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#Create MoE model and train it with two experts
moeModel = Mixture(x_train, y_train, x_test, y_test, experts, inputs, sc)
moeModel.train_init(datagen, moe_weights_file)


