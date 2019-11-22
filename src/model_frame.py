from keras.layers import Input

class ModelFrame(object):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.inputs = Input(shape=x_train.shape[1:])
        print("---------------------")
        print(self.inputs)
        print("---------------------")

        self.model = None
        