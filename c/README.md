# NanoNN in C

This implementation is based on C89 standard and uses no memory allocations. It is a single-header library that you can simply drop into your code base. Compatible with C++ as well.

## Usage

```
#include "nn.h"

...
// A network is an array of layers, last layer must be NN_EMPTY.
struct nn_layer nn[] = {
  NN_DENSE(3, 2, NN_FLAG_RELU), // 3 inputs, 2 units
  NN_DENSE(2, 1, NN_FLAG_SIGMOID), // 2 inputs, 1 output (=1 unit)
  NN_EMPTY(), 
};

// How much memory does a network use
size_t memsz = nn_init(nn, NULL, 0);
// Allocate memory
void *mem = malloc(memsz);
// Initialize network
nn_init(nn, mem, memsz);
// Initialize weights, if needed
nn[0].weights.data[0] = 1.74481176;

// Some input vector
float *x = ....;
// Ask network to predict the output
float *z = nn_predict(nn, x);

// Or run a single training iteration:
float learning_rate = 0.5; // how fast to change the weights
float err = nn_train(nn, x, y, learning_rate);
```

Activation functions (LINEAR, RELU, LRELU, SIGMOID, SOFTMAX) and optional bias are controlled through the bitwise flag set.

## Custom layer types

A layer is a structure with pre-defined fields, but the behavior of the layer is controlled by two function pointers - `forward` and `backward`. The first one is used to predict the output by the input, the second to adjust the weights during the training. It is fine to leave the backward propagation function empty if you don't expect the network to be trained.

Each layer has 4 pre-defined vectors inside: inputs, outputs, errors and weights (with biases as the right-most column). The last vector is cache, in dense layers it's empty, but other implementations may store arbitrary numeric data there.
