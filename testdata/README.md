# Test data

This folder contains datasets and scripts to generate test data:

* `iris.csv` - a well-known [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set). The first four columns are features (length and width of sepals and petals). The last 3 columns are species, one-hot encoding (i.e. "0,1,0"  means versicolor, "1,0,0" is setosa and "0,0,1" is virginica).
* `tf_act.py` - a script to check activation functions and their derivative.
* `tf_forward.py` - a script to check one unit forward propagation.
* `tf_backward.py` - a script to check backwards propagation of a 2-layer full-connected network.
* `tf_iris.py` - a script to solve Iris problem, and export weights.
* `tf_mnist.py` - a script to prepare MNIST dataset, train a model and export weights.

Please, use the values from `tf_act.py`, `tf_forward.py` and `tf_backward.py` to ensure that your implementation works correctly. You might want to test it on the Iris dataset. Also, you might want to import trained weights from the MNIST model to see how to import data from Tensorflow into your model.
