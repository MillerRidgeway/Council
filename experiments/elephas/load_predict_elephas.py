from __future__ import absolute_import
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

from pyspark import SparkContext, SparkConf


# Define basic parameters
batch_size = 64
nb_classes = 10
epochs = 1

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

#Model loading
json_model = open('mnist_test.json', 'r')
loaded_model_json = json_model.read()
json_model.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('mnist_test.h5')

#Make sure model was loaded in correctly (should be 9.80% accuracy)
sgd = SGD(lr=0.1)
loaded_model.compile(sgd, 'categorical_crossentropy', ['acc'])
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

# Create Spark context
conf = SparkConf().setAppName('Mnist_Spark_MLP').setMaster('local[8]')
sc = SparkContext(conf=conf)

# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, x_train, y_train)

# Initialize SparkModel from Keras model and Spark context
spark_model = SparkModel(loaded_model, frequency='epoch', mode='asynchronous')

# Evaluate Spark model by evaluating the underlying model
score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', score[1])