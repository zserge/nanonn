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

// Neural network is a sequence of layers. It is represented as an array with
// two additional methods, predict and train.
export const NN = (...layers) => {
  const nn = layers;
  // Predict returns a predicted outputs vector for the given input vector x.
  nn.predict = x => nn.reduce((x, l) => l.forward(x), x);
  // Train performs a single iteration of backpropagation and returns an
  // evaluation error.
  nn.train = (x, y, rate = 1) => {
    const inputs = [x, ...nn.map(l => (x = l.forward(x)))];
    const errors = y.map((yi, i) => yi - x[i]);
    nn.reduceRight((e, l, i) => l.backward(inputs[i], e, rate), errors);
    return errors.reduce((sum, ei) => sum + ei * ei, 0) / y.length;
  };
  return nn;
};

// Sigmoid activation function and its derivative
export const sigm = {
  f: x => 1 / (1 + Math.exp(-x)),
  df: x => x * (1 - x),
};

// ReLU activation function and its derivative
export const relu = {
  f: x => (x > 0 ? x : 0),
  df: x => (x > 0 ? 1 : 0),
};

// Dense returns a fully connected dense layer
export const Dense = ({units = 1, inputs = 1, act = sigm, bias = true, weights}) => {
  const N = inputs + 1;
  const w = Array(units * N).fill(0); // weights and biases
  const e = Array(inputs).fill(0); // errors to return to the previous layer
  const z = Array(units).fill(0); // outputs to pass to the next layer
	w.forEach((_, i) => (w[i] = weights ? weights[i] : Math.random()));
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
