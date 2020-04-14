# NanoNN in JavaScript

This implementation is a tiny self-contained ES6 module. The goal is to keep it as small as possible, but still somewhat useful.

## Usage

ES6 modules are supported by both, decent browsers and latest NodeJS versions:

```
import {NN, Dense, sigmoid} from './nn.js';
// Create network with one hidden layer of 3 units (neurons)
const network = NN(
	Dense({inputs: 2, units: 3, act: sigmoid}),
	Dense({inputs: 3, units: 1, act: sigmoid}),
);

// Predict output by input
const input = [0, 1];
const output = network.predict(input);

// Or train a network by giving the expected output:
const error = network.train(input, expectedOutput, learningRate);

// Network and dense layer are vectors, so you can manipulate, store/load weights easily:
const s = JSON.stringify(network); // save all weigths to JSON string
network[0][1] = 0.5; // set weights individually
```

## Customizations

Dense layer can be customized by using various activation functions, enabling/disabling bias. Custom layer types can be implemented as well. A layer must be an object with two functions (methods) - `forward(inputVector, network)` and `backward(inputVector, errorVector, learningRate, network)`. See how Dense layer is implemented for more details.
