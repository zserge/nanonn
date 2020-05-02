#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

X = tf.constant([[0.05, 0.1]])
Y = tf.constant([[0.01, 0.99]])

W1 = tf.constant_initializer([0.15, 0.25, 0.2, 0.3])
B1 = tf.constant_initializer([0.35, 0.35])
W2 = tf.constant_initializer([0.4, 0.5, 0.45, 0.55])
B2 = tf.constant_initializer([0.6, 0.6])

model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Dense(
        2,
        activation="sigmoid",
        kernel_initializer=W1,
        bias_initializer=B1,
        input_shape=(2,),
    )
)
model.add(
    tf.keras.layers.Dense(
        2, activation="sigmoid", kernel_initializer=W2, bias_initializer=B2
    )
)
model.build()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)

Z = model(X)
print("Predicted:", Z[0].numpy())

loss_fn = lambda: tf.keras.losses.mse(model(X), Y)
var_list_fn = lambda: model.trainable_weights
optimizer.minimize(loss_fn, var_list_fn)

print("W1:", model.layers[0].get_weights()[0])
print("B1:", model.layers[0].get_weights()[1])
print("W2:", model.layers[1].get_weights()[0])
print("B2:", model.layers[1].get_weights()[1])
