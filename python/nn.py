# Copyright 2020 Sergii Zaitsev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import exp, sqrt, log
from random import uniform

sigmoid = (lambda x: 1 / (1 + exp(-x)), lambda x: x * (1 - x))
relu = (lambda x: x * (x > 0), lambda x: 1.0 * (x > 0))
lrelu = (lambda x: x if x > 0 else 0.01 * x, lambda x: 1 if x > 0 else 0.01)
softplus = (lambda x: log(1.0 + exp(x)), lambda x: 1.0 / (1.0 + exp(-x)))
linear = (lambda x: x, lambda x: 1)


class NN:
    def __init__(self, *args, layers=None):
        self.layers = layers or args

    def __getitem__(self, index):
        return self.layers[index]

    def predict(self, x):
        for l in self.layers:
            x = l(self, x)
        return x

    def train(self, x, y, rate=1):
        inputs = [0 for l in self.layers]
        for i, l in enumerate(self.layers):
            inputs[i] = x
            x = l(self, x, True)
        e = 0
        errors = [0 for i in range(len(y))]
        for i in range(len(y)):
            errors[i] = y[i] - x[i]
            e += errors[i] * errors[i]
        for i in range(len(self.layers) - 1, -1, -1):
            errors = self.layers[i].backward(self, inputs[i], errors, rate)
        return e / len(y)


class Dense:
    def __init__(
        self, units=1, inputs=1, use_bias=True, activation=sigmoid, weights=None,
    ):
        self.units = units
        self.inputs = inputs
        self.use_bias = use_bias
        self.activation = activation
        self.weights = weights
        # He et al random weight initialization
        weights_range = sqrt(2 / inputs)
        if not self.weights:
            self.weights = [
                uniform(-weights_range, weights_range)
                for i in range(units * (inputs + 1))
            ]
        self.outputs = [0 for i in range(units)]
        self.errors = [0 for i in range(inputs)]

    def set_weights(self, weights):
        self.weights = weights

    def get_weights(self):
        return self.weights

    def __call__(self, nn, x, training=False):
        N = self.inputs + 1
        for i in range(self.units):
            w = 0
            for j in range(self.inputs):
                w = w + x[j] * self.weights[i * N + j]
            if self.use_bias:
                self.outputs[i] = self.activation[0](w + self.weights[i * N + N - 1])
        return self.outputs

    def backward(self, nn, x, e, rate):
        N = self.inputs + 1
        df = self.activation[1]
        for j in range(self.inputs):
            self.errors[j] = 0
            for i in range(self.units):
                self.errors[j] += e[i] * df(self.outputs[i]) * self.weights[i * N + j]
        for i in range(self.units):
            for j in range(self.inputs):
                self.weights[i * N + j] += rate * e[i] * df(self.outputs[i]) * x[j]
            if self.use_bias:
                self.weights[i * N + N - 1] += rate * e[i] * df(self.outputs[i])
        return self.errors
