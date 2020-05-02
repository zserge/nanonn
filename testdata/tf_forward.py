#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

W = tf.constant([[1.74481176], [-0.7612069], [0.3190391]])
B = tf.constant([-0.24937038])


@tf.function
def perceptron(X, W, B):
    layer = tf.add(tf.matmul(X, W), B)
    act = tf.nn.sigmoid(layer)
    return act


X1 = tf.constant([[1.62434536, -0.52817175, 0.86540763]])
Y1 = perceptron(X1, W, B)
print(Y1[0][0].numpy())

X2 = tf.constant([[-0.61175641, -1.07296862, -2.3015387]])
Y2 = perceptron(X2, W, B)
print(Y2[0][0].numpy())
