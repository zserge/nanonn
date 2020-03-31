from math import exp
from random import uniform

sigmoid = (lambda x: 1 / (1 + exp(-x)), lambda x: x * (1 - x))


class NN:
    def __init__(self, *args, layers=None):
        self.layers = layers or args

    def __getitem__(self, index):
        return self.layers[index]

    def predict(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def train(self, x, y, rate=1):
        inputs = [0 for l in self.layers]
        for i, l in enumerate(self.layers):
            inputs[i] = x
            x = l(x)
        e = 0
        errors = [0 for i in range(len(y))]
        for i in range(len(y)):
            errors[i] = y[i] - x[i]
            e += errors[i] * errors[i]
        for i in range(len(self.layers) - 1, 0, -1):
            errors = self.layers[i].backward(inputs[i], errors, rate)
        return e / len(y)


class Dense:
    def __init__(
        self, units=1, inputs=1, use_bias=True, activation=sigmoid, weights=None,
    ):
        self.units = units
        self.inputs = inputs
        self.activation = activation
        self.weights = weights
        if not self.weights:
            self.weights = [uniform(-1, 1) for i in range(units * (inputs + 1))]
        self.outputs = [0 for i in range(units)]
        self.errors = [0 for i in range(inputs)]

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def __call__(self, x):
        N = self.inputs + 1
        for i in range(self.units):
            w = 0
            for j in range(self.inputs):
                w = w + x[j] * self.weights[i * N + j]
            self.outputs[i] = self.activation[0](w + self.weights[i * N + N - 1])
        return self.outputs

    def backward(self, x, e, rate=1):
        N = self.inputs + 1
        df = self.activation[1]
        for j in range(self.inputs):
            self.errors[j] = 0
            for i in range(self.units):
                self.errors[j] += e[i] * df(self.outputs[i]) * self.weights[i * N + j]
        for i in range(self.units):
            for j in range(self.inputs):
                self.weights[i * N + j] += rate * e[i] * df(self.outputs[i]) * x[j]
            self.weights[i * N + N - 1] += rate * e[i] * df(self.outputs[i])
        return self.errors
