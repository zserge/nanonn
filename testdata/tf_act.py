#!/usr/bin/env python3

import numpy as np

X = np.linspace(-1, 1, 9)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1 - y)


def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_derivative(y):
    return 1 - y ** 2


def relu(x):
    return np.where(x >= 0, x, 0)


def relu_derivative(y):
    return np.where(y >= 0, 1, 0)


def leaky_relu(x, a=0.01):
    return np.where(x >= 0, x, a * x)


def leaky_relu_derivative(x, a=0.01):
    return np.where(x >= 0, 1, a)


print()
print("    X      sigm(X)     sigm'(X)")
for x in X:
    print("%5.2f %12f %12f" % (x, sigmoid(x), sigmoid_derivative(x)))

print()
print("    X      tanh(X)     tanh'(X)")
for x in X:
    print("%5.2f %12f %12f" % (x, tanh(x), tanh_derivative(x)))

print()
print("    X      relu(X)     relu'(X)")
for x in X:
    print("%5.2f %12f %12f" % (x, relu(x), relu_derivative(x)))

print()
print("    X     lrelu(X)     lrelu'(X)")
for x in X:
    print("%5.2f %12f %12f" % (x, leaky_relu(x), leaky_relu_derivative(x)))
