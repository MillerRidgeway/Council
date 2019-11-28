from keras.layers import Input
from elephas.utils.rdd_utils import to_simple_rdd

class ModelFrame(object):
    def __init__(self, x_train, y_train, x_test, y_test, spark_context = None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        if(spark_context is not None):
            self.rdd = to_simple_rdd(spark_context, x_train, y_train)
        else:
            self.rdd = None
        