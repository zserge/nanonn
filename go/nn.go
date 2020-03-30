// Copyright 2020 Sergii Zaitsev
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package nn is a very simple implementation of a basic neural network with
// hidden layers. Out of the box it comes with a fully connected dense layer
// implementation, but it can be extended with custom layer types if needed.
package nn

import (
	"fmt"
	"math"
	"math/rand"
)

// Network is a sequence of layers.
type Network struct {
	layers []Layer     // sequence of layers
	inputs [][]float64 // cache of input vectors passed between the layers
	errors []float64   // errors vector for the output layer
}

// New returns a new sequential network constructed from the given layers. An
// error is returned if the number of inputs and outputs in two adjacent layers
// is not the same.
func New(layers ...Layer) (Network, error) {
	for i, l := range layers {
		if i > 0 {
			if a, b := l.Inputs(), layers[i-1].Outputs(); a != b {
				return Network{}, fmt.Errorf("expected %d inputs, got %d", a, b)
			}
		}
	}
	n := layers[len(layers)-1].Outputs()
	errors := make([]float64, n, n)
	inputs := make([][]float64, len(layers), len(layers))
	return Network{layers: layers, inputs: inputs, errors: errors}, nil
}

// Predict passes input vector x into the network and returns the output vector.
func (n Network) Predict(x []float64) []float64 {
	for _, l := range n.layers {
		x = l.Forward(x)
	}
	return x
}

// Train runs one backpropagation iteration through the network. It takes input
// vector x and expected output vector y, also a learning rate parameter.
func (n Network) Train(x, y []float64, rate float64) float64 {
	for i, l := range n.layers {
		n.inputs[i] = x
		x = l.Forward(x)
	}
	e := 0.0
	for i := 0; i < len(y); i++ {
		n.errors[i] = y[i] - x[i]
		e += n.errors[i] * n.errors[i]
	}
	errors := n.errors
	for i := len(n.layers) - 1; i >= 0; i-- {
		errors = n.layers[i].Backward(n.inputs[i], errors, rate)
	}
	return e / float64(len(y))
}

// Layer is a singe network layer. It is an interface, so various layer types
// can be implemented layer. Layer must return a number of its inputs and
// outputs. Also it must implement two methods - forward and backpropagation.
type Layer interface {
	Inputs() int
	Outputs() int
	Weights() []float64
	SetWeights(w []float64)
	Forward(input []float64) []float64
	Backward(input, errors []float64, rate float64) []float64
}

// Sigmoid activation function and its derivation.
func sigm(x float64) float64  { return 1 / (1 + math.Exp(-x)) }
func dsigm(x float64) float64 { return x * (1 - x) }

// Default implementation of a fully-connected layer.
type dense struct {
	weights []float64
	outputs []float64
	errors  []float64
}

// Dense returns a new dense fully-connected layer with sigmoid activation function and the given number of inputs and neurons.
func Dense(units, inputs int) Layer {
	l := &dense{
		weights: make([]float64, units*(inputs+1), units*(inputs+1)),
		outputs: make([]float64, units, units),
		errors:  make([]float64, inputs, inputs),
	}
	// Initialize weights randomly
	for i := range l.weights {
		l.weights[i] = rand.Float64()*2 - 1
	}
	return l
}

func (l *dense) Outputs() int           { return len(l.outputs) }
func (l *dense) Inputs() int            { return len(l.errors) }
func (l *dense) Weights() []float64     { return l.weights }
func (l *dense) SetWeights(w []float64) { copy(l.weights, w) }

func (l *dense) Forward(x []float64) []float64 {
	N := l.Inputs() + 1
	for i := 0; i < l.Outputs(); i++ {
		sum := float64(0)
		for j := 0; j < l.Inputs(); j++ {
			sum = sum + x[j]*l.weights[i*N+j]
		}
		l.outputs[i] = sigm(sum + l.weights[i*N+N-1])
	}
	return l.outputs
}

func (l *dense) Backward(x, e []float64, rate float64) []float64 {
	N := l.Inputs() + 1
	for j := 0; j < l.Inputs(); j++ {
		l.errors[j] = 0
		for i := 0; i < l.Outputs(); i++ {
			l.errors[j] += e[i] * dsigm(l.outputs[i]) * l.weights[i*N+j]
		}
	}
	for i := 0; i < l.Outputs(); i++ {
		for j := 0; j < l.Inputs(); j++ {
			l.weights[i*N+j] += rate * e[i] * dsigm(l.outputs[i]) * x[j]
		}
		l.weights[i*N+N-1] += rate * e[i] * dsigm(l.outputs[i])
	}
	return l.errors
}
