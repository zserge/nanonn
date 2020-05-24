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
//

/**
 * Predicts output vector by the given input vector.
 * @name PredictFunction
 * @function
 * @param {...Number} [inputs] input vector.
 * @returns {...Number} output vector.
 */

/**
 * Trains layers to fit the expected output vector better.
 * @name TrainFunction
 * @function
 * @param {...Number} [inputs] input vector.
 * @param {...Number} [outputs] expected outputs vector.
 * @param {Number} [rate] learning rate, use 0 to only calculate the error.
 * @returns {Number} prediction error.
 */

/**
 * A sequential neural network model.
 * @typedef NN
 * @type {object}
 * @property {PredictFunction} predict Predict output by the given input vector.
 * @property {function} train Adjust network layers to fit the expected output vector.
 */

/**
 * Creates a neural network made of a sequence of layers. The network is
 * represented as an array of layers with two additional methods, predict and
 * train.
 *
 * @function
 * @param {...Layer} [layers] Variadic list of layers.
 * @returns {NN} A neural network model.
 *
 * @example
 * const nn = NN(Dense({units: 10, inputs: 4}), Dense({inputs: 10, units: 2}));
 */
export const NN = (...layers) => {
  const nn = layers;
  // Cost function, mean squared error.
  nn.cost = (x, y) => {
    const errors = y.map((yi, i) => yi - x[i]);
    return [errors.reduce((sum, ei) => sum + ei * ei, 0) / y.length, ...errors];
  };
  // Predict returns a predicted outputs vector for the given input vector x.
  nn.predict = x => nn.reduce((x, l) => l.forward(x, nn, false), x);
  // Train performs a single iteration of backpropagation and returns an
  // evaluation error.
  nn.train = (x, y, rate = 1) => {
    const inputs = [x, ...nn.map(l => (x = l.forward(x, nn, true)))];
    const [totalError, ...errors] = nn.cost(x, y);
    nn.reduceRight((e, l, i) => l.backward(inputs[i], e, rate, nn), errors);
    return totalError;
  };
  return nn;
};

/** Sigmoid activation function and its derivative */
export const sigmoid = {
  f: x => 1 / (1 + Math.exp(-x)),
  df: x => x * (1 - x),
};

/** ReLU activation function */
export const relu = {
  f: x => (x > 0 ? x : 0),
  df: x => (x > 0 ? 1 : 0),
};

/** Leaky ReLU activation function */
export const lrelu = {
  f: x => (x > 0 ? x : 0.01 * x),
  df: x => (x > 0 ? 1 : 0.01),
};

/** Linear activation function */
export const linear = {
  f: x => x,
  df: () => 1.0,
};

/** SoftPlus activation function */
export const softplus = {
  f: x => Math.log(1.0 + Math.exp(x)),
  df: x => 1.0 / (1.0 + Math.exp(-x)),
};

/**
 * Returns a fully connected dense layer.
 * @function
 * @param options Layer configuration.
 * @param options.inputs {Number} Number of layer inputs. Must match the number
 * of units in the previous layer, or the size of the input vector.
 * @param options.units {Number} Number of units (neurons).
 * @param [options.act] {ActivationFunction} Activation function.
 * @param [options.bias] {Boolean} Add trainable bias value. True by default.
 * @param [options.weights] {Array.<Number>} Optional weights matrix. Initialized randomly by default.
 * @returns Layer A dense layer object.
 */
export const Dense = ({
  units = 1,
  inputs = 1,
  act = sigmoid,
  bias = true,
  weights,
}) => {
  const N = inputs + 1;
  const w = Array(units * N).fill(0); // weights and biases
  const e = Array(inputs).fill(0); // errors to return to the previous layer
  const z = Array(units).fill(0); // outputs to pass to the next layer
  w.forEach((_, i) => (w[i] = weights ? weights[i] : Math.random() / 100));
  w.forward = x => {
    z.forEach((_, i) => {
      let sum = x.reduce((sum, xj, j) => sum + xj * w[i * N + j], 0);
      z[i] = act.f(sum + (bias ? w[i * N + N - 1] : 0));
    });
    return z;
  };
  w.backward = (x, err, rate) => {
    e.forEach(
      (_, j) =>
        (e[j] = err.reduce(
          (sum, ei, i) => sum + ei * act.df(z[i]) * w[i * N + j],
          0,
        )),
    );
    z.forEach((zi, i) => {
      x.forEach((xj, j) => (w[i * N + j] += rate * err[i] * act.df(zi) * xj));
      if (bias) {
        w[i * N + N - 1] += rate * err[i] * act.df(zi);
      }
    });
    return e;
  };
  return w;
};
