import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def l1_loss(y_pred, y_true):
    return np.sum((y_pred - y_true), axis=0)


def cross_entropy_loss(y_pred, y_true, epsilon=1e-12, as_num=False):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    N = y_pred.shape[0]
    ce = -np.sum(y_true * np.log(y_pred + 1e-9)) / N
    if as_num:
        return ce.ravel()[0]
    return ce


LOSS_DICT = {
    "l1_loss": l1_loss,
    "ce_loss": cross_entropy_loss
}


class Activation:

    @staticmethod
    def activation_function(x):
        raise NotImplementedError('To be implemented by a subclass')

    @staticmethod
    def activation_derivative(x):
        raise NotImplementedError('To be implemented by a subclass')


class Sigmoid(Activation):

    @staticmethod
    def activation_function(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def activation_derivative(x):
        return Sigmoid.activation_function(x) * (1 - Sigmoid.activation_function(x))


class Relu(Activation):

    @staticmethod
    def activation_function(x):
        return x * (x > 0)

    @staticmethod
    def activation_derivative(x):
        return 1 * (x > 0)


class Softmax(Activation):
    @staticmethod
    def activation_function(x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    @staticmethod
    def activation_derivative(x):
        return x


ACTIVATION_DICT = {
    "sigmoid": Sigmoid,
    "relu": Relu,
    "softmax":Softmax
}


class Optimizer:

    @staticmethod
    def update(weights, bias, lr, de_do, de_dw, s_dw, s_db):
        raise NotImplementedError('To be implemented by a subclass')


class GradientDescent(Optimizer):

    @staticmethod
    def update(weights, bias, lr, de_do, de_dw, s_dw, s_db):
        weights = weights - lr * de_dw
        bias = bias - lr * de_do
        return weights, bias, s_dw, s_db


class RMSProp(Optimizer):

    @staticmethod
    def update(weights, bias, lr, de_do, de_dw, s_dw, s_db):
        s_dw = 0.8 * s_dw + 0.2 * np.square(de_dw)
        s_db = 0.8 * s_db + 0.2 * np.square(de_do)
        weights = weights - lr * np.divide(de_dw, np.sqrt(s_dw)+1e-15)
        bias = bias - lr * np.divide(de_do, np.sqrt(s_db)+1e-15)
        return weights, bias, s_dw, s_db


OPTIMIZER_DICT = {
    "grad_d": GradientDescent,
    "rmsprop": RMSProp
}


class DenseLayer:

    def __init__(self, input_dims, units, activation='relu'):
        self.input_dims = input_dims
        self.units = units
        self.activation = ACTIVATION_DICT[activation]
        self.weights = np.random.random(size=(input_dims, units))
        self.bias = np.random.random(size=(units,))
        self.current_input = None
        self.s_dw = float(0)
        self.s_db = float(0)

    def forward(self, x):
        self.current_input = x
        out = np.matmul(x, self.weights) + self.bias
        out = self.activation.activation_function(out)
        return out

    def backward(self, de_do, lr, optimizer):
        inputs = np.sum(self.current_input, axis=0)  # shape (input_dim, )
        de_do = de_do * self.activation.activation_derivative(np.matmul(inputs, self.weights) + self.bias)
        de_dw = np.outer(inputs, de_do)  # shape (input_dim, units)
        self.weights, self.bias, self.s_dw, self.s_db = optimizer.update(self.weights, self.bias,
                                                                              lr, de_do, de_dw, self.s_dw, self.s_db)
        de_di = np.dot(de_do, self.weights.T)
        return de_di


class Network:

    def __init__(self, input_dims, loss_function='l1_loss', optimizer='rmsprop'):
        self.input_dims = input_dims
        self.out_dims = None
        self.loss_function = LOSS_DICT[loss_function]
        self.optimizer = OPTIMIZER_DICT[optimizer]
        self.layers = []

    def add_layer(self, layer):
        if len(self.layers) > 0:
            assert self.layers[-1].units == layer.input_dims, "invalid layer shape"
            assert self.layers[-1].activation != 'softmax', "Softmax activation is valid for output layer only"
        else:
            assert self.input_dims == layer.input_dims, "invalid layer shape"
        self.layers.append(layer)

    def forward_prop(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        if self.out_dims is None:
            self.out_dims = out.shape[-1]
        return out

    def back_prop(self, x, y_true, lr):
        y_pred = self.forward_prop(x)
        de_dy = self.loss_function(y_pred, y_true)
        de_do = de_dy
        for layer in reversed(self.layers):
            if len(de_do.shape) > 1:
                de_do = np.sum(de_do, axis=0)
            de_di = layer.backward(de_do, lr, self.optimizer)
            de_do = de_di

    def train(self, x, y, epochs=500, batch_size=4, interval=50, lr=0.001):
        for i in range(epochs):
            assert x.shape[0] == y.shape[0], 'invalid data arrays'
            for batch_id in range(0, x.shape[0], batch_size):
                x_, y_ = x[batch_id:batch_id + batch_size, :], y[batch_id:batch_id + batch_size]
                assert x_.shape[0] == y_.shape[0], 'invalid data arrays'
                self.back_prop(x_, y_, lr)

            if (i+1) % interval == 0:
                y_p = self.forward_prop(x)
                print("epoch {} - train accuracy: {:.2f}".format(
                    (i+1),
                    accuracy_score(np.argmax(y, axis=1), np.argmax(y_p, axis=1))
                ))

    def predict(self, x):
        out = self.forward_prop(x)
        return np.argmax(out, axis=1)


########################################################################################################################

from sklearn import datasets

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, shuffle=True)

layer1 = DenseLayer(input_dims=4, units=10, activation='relu')
layer2 = DenseLayer(input_dims=10, units=3, activation='softmax')

model = Network(input_dims=4, loss_function='l1_loss', optimizer='rmsprop')

model.add_layer(layer1)
model.add_layer(layer2)

y_train_hot = np.zeros((len(y_train), len(np.unique(y_train))))
y_train_hot[np.arange(len(y_train)), y_train] = 1

model.train(x_train, y_train_hot, epochs=500, batch_size=4, interval=50, lr=0.001)

y_pred = model.predict(x_test)

test_acc = accuracy_score(y_true=y_test, y_pred=y_pred)

print("test accuracy: {:.2f}".format(test_acc))

########################################################################################################################

# from keras.datasets import cifar10
#
# (x_Train, y_Train), (x_Test, y_Test) = cifar10.load_data()
# x_train, x_test, y_train, y_test = train_test_split(x_Test, y_Test, test_size=0.99, shuffle=True)
# x_train = np.reshape(x_train, (x_train.shape[0], (32*32*3)))
# y_train = np.squeeze(y_train)
#
#
# layer1 = DenseLayer(input_dims=3072, units=6000, activation='relu')
# layer2 = DenseLayer(input_dims=6000, units=3000, activation='relu')
# layer3 = DenseLayer(input_dims=3000, units=1000, activation='relu')
# layer4 = DenseLayer(input_dims=1000, units=125, activation='relu')
# layer5 = DenseLayer(input_dims=125, units=10, activation='softmax')
#
# model = Network(input_dims=3072)
#
# model.add_layer(layer1)
# model.add_layer(layer2)
# model.add_layer(layer3)
# model.add_layer(layer4)
# model.add_layer(layer5)
#
# y_train_hot = np.zeros((len(y_train), len(np.unique(y_train))))
# y_train_hot[np.arange(len(y_train)), y_train] = 1
#
# model.train(x_train, y_train_hot, epochs=10, batch_size=10, interval=1, lr=0.005)
#
# y_pred = model.predict(x_train)
#
# test_acc = accuracy_score(y_true=y_train, y_pred=y_pred)
#
# print("test accuracy: {:.2f}".format(test_acc))
