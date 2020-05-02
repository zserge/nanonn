#!/usr/bin/env python3

import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
import random

dataset = []
with open("iris.csv") as iris:
    entries = csv.reader(iris, delimiter=",")
    for entry in entries:
        dataset.append([float(field) for field in entry])

dataset = np.array(dataset)
np.random.shuffle(dataset)

train_x, train_y = dataset[:120, [0, 1, 2, 3]], dataset[:120, [4, 5, 6]]
test_x, test_y = dataset[120:, [0, 1, 2, 3]], dataset[120:, [4, 5, 6]]

model = tf.keras.models.Sequential()
model.add(Dense(10, activation="relu", input_shape=(4,)))
model.add(Dense(10, activation="relu"))
model.add(Dense(3, activation="sigmoid"))

optimizer = tf.keras.optimizers.SGD(lr=0.01)
model.compile(optimizer, loss="mse", metrics=["accuracy"])

model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)
model.evaluate(test_x, test_y)
