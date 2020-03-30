# nanonn

[![go.dev reference](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white&style=flat-square)](https://pkg.go.dev/github.com/zserge/nanonn/go)
[![Go Report Card](https://goreportcard.com/badge/github.com/zserge/nanonn)](https://goreportcard.com/report/github.com/zserge/nanonn)

NanoNN is a nano-framework for neural networks. Or, if you wish, a collection of toy neural network implementations in different programming languages.

In no sense it is a replacement for Tensorflow or PyTorch, but you might find this project useful for hobby projects, embedded systems and for didactic purposes. Nothing teaches better than a toy code one can play with.

# Features

* Implements a sequential multi-layer neural network.
* A fully-connected dense layer is provided.
* Can be extended by implementing other layer types manually.
* Small, typical implenentation is around 100 lines of code.
* Zero dependencies.
* Covered with tests and benchmarks.
* Simplicity and correctness is often more preferred than additional features or highly optimized performance.

# Notation

* N - number of inputs.
* M - number of outputs.
* X - input vector of N elements.
* Y - expected output vector of M elements.
* Z - actual output vector of M elements.
* W - weight matrix of N×M elements, or (N+1)×M elements when bias is enabled.
* E - errors vector of M elements used in backpropagation.
* E' - errors vector for the previous layer, of N elements.
* D - delta vector of M elements used in backpropagation.
* sigm - activation function, typically sigmoid, ReLU or SoftMax.
* dsigm - partially derivated activation function.
* rate - learning rate.

# Implementation

`Network` is simply a sequence of `Layers`. Layers are assumed to be sequential. During forward propagation ouptut vector from the previous layer is passed into the next layer. During backpropagation an input and an error vector from the one layer is passed into the prevous layer.

`Layer` is an interface, to allow developers plug in their own layer types, if needed. Each layer consist of an array of units ("neurons"), and has a fixed number of inputs and outputs. The output shape of one layer should match the input shape of the following layer. Layers should allow to set and get weights matrix so that it would be possible to save and restore network state.

`Dense` layer is a fully-connected layer of units, where outputs of the previous layer are connected to each and every unit in the current layer.

In a `Dense` layer units are represented as a weight matrix of M×(N+1) dimensions, where `N` is a number of inputs and `M` is a number of outputs of the layer. One is added to the number of outputs to keep bias value in the same weight matrix for simplicity. Most implementations will keep weight matrix as a linear array, in this case each "row" represents the weights of a single unit (including bias):

Forward propagation in a dense layer happens trivially, for each unit we multiply inputs to the weights and sum them up, then activation function `act` (typically, sigmoid, ReLU or SoftMax) is taken from that value, forming the output. To predict output from the input, the network calls forward propagation in each layer sequentially passing output vector from one layer to another:

Z = act(W·X + W<sub>bias<sub>), or

```
for i = 0..M
	sum = 0
	for j = 0..N
		sum = sum + x[j]*w[i,j]
	z[i] = sigm(sum + w[i,N+1])
```

Backpropagation is a little bit more tricky. Each layer should adjust its weights and return the error vector to help previous layers correct their weight too. For the output layer the error vector is calculated as a different between the expected and the actual outputs, E = Z - Y.

In each layer a delta vector is calculated, D = E * dsigm(Z).

Then the error vector for the previous layer is calculated as E' = D·W.

Finally, the weights are adjusted, W = W + X·D.

This procedure is repeated from the output layer to the input layer.

```
for j = 0..N
	E'[j] = 0
	for i = 0..M
		E'[j] = E'[j] + E[i]*dsigm(Z[i] * W[i,j])
for i = 0..M
	for j = 0..N
		W[i,j] = W[i,j] + rate * E[i] * dsigm(Z[i]*X[j])
	W[i,N+1] = W[i,N+1] + rate * E[i] * dsigm(Z[i])
```

## Contributing

Pull requests are welcome. For new features or major changes, please open an issue first to discuss what you would like to change.

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
