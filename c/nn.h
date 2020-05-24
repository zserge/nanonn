/*
 * Copyright 2020 Sergii Zaitsev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NN_H
#define NN_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct nn_vec {
  size_t len;
  float *data;
};

#define NN_FLAG_NO_BIAS 1

#define NN_FLAG_LINEAR (0 << 1u)
#define NN_FLAG_RELU (1u << 1u)
#define NN_FLAG_LRELU (2u << 1u)
#define NN_FLAG_SIGMOID (3u << 1u)
#define NN_FLAG_SOFTPLUS (4u << 1u)
#define NN_ACTFN (3u << 1u)

struct nn_layer {
  unsigned int flags;
  struct nn_vec input;
  struct nn_vec output;
  struct nn_vec weights;
  struct nn_vec errors;
  struct nn_vec cache;
  void (*forward)(struct nn_layer *l, int training);
  void (*backward)(struct nn_layer *l, float *e, float rate);
};

#define NN_LAYER_EMPTY(l) ((l)->input.len * (l)->output.len == 0)

#define NN_EMPTY()                                                             \
  { 0 }

#define NN_DENSE(out, in, flags)                                               \
  {                                                                            \
    (flags), {(out), NULL}, {(in), NULL}, {(((out) + 1) * (in)), NULL},        \
	{(out), NULL}, {0, NULL}, nn_dense_forward, nn_dense_backward          \
  }

void nn_print(struct nn_layer *layers) {
  unsigned int i, j;
  struct nn_layer *l = layers;
  for (i = 0; !NN_LAYER_EMPTY(l); l++, i++) {
    printf("LAYER %d\n", i);
    printf("  X (%p): ", (void *)l->input.data);
    if (l->input.data != NULL) {
      for (j = 0; j < l->input.len; j++) {
	printf("%.02f ", l->input.data[j]);
      }
    }
    printf("\n");
    printf("  Y (%p): ", (void *)l->output.data);
    for (j = 0; j < l->output.len; j++) {
      printf("%.02f ", l->output.data[j]);
    }
    printf("\n");
    printf("  W (%p): ", (void *)l->weights.data);
    for (j = 0; j < l->weights.len; j++) {
      printf("%.02f ", l->weights.data[j]);
    }
    printf("\n");
    printf("  E (%p): ", (void *)l->errors.data);
    for (j = 0; j < l->errors.len; j++) {
      printf("%.02f ", l->errors.data[j]);
    }
    printf("\n");
    printf("  C (%p): ", (void *)l->cache.data);
    for (j = 0; j < l->cache.len; j++) {
      printf("%.02f ", l->cache.data[j]);
    }
    printf("\n");
  }
}

size_t nn_init(struct nn_layer *layers, const void *mem, size_t memsz) {
  size_t expected = 0;
  size_t last_outputs = 0;
  float *prev_input = NULL;
  struct nn_layer *l = layers;
  struct nn_vec vec = {0, NULL};
  vec.data = (float *)mem;
  vec.len = memsz / sizeof(float);
  for (; !NN_LAYER_EMPTY(l); l++) {
    /* Remember number of outputs, network needs to store errors for the last
     * layer in a vector of the same size */
    last_outputs = l->output.len;
    /* Connect layer input */
    l->input.data = prev_input;
    /* Connect layer output */
    if (expected + l->output.len < vec.len) {
      l->output.data = vec.data + expected;
      prev_input = l->output.data;
    }
    expected = expected + l->output.len;
    /* Prepare layer weights */
    if (expected + l->weights.len < vec.len) {
      l->weights.data = vec.data + expected;
    }
    expected = expected + l->weights.len;
    /* Prepare layer errors */
    if (expected + l->errors.len < vec.len) {
      l->errors.data = vec.data + expected;
    }
    expected = expected + l->errors.len;
    /* Prepare layer cache */
    if (expected + l->cache.len < vec.len) {
      l->cache.data = vec.data + expected;
    }
    expected = expected + l->cache.len;
  }
  expected = expected + last_outputs;
  return expected * sizeof(float);
}

float *nn_predict(struct nn_layer *layers, float *x) {
  struct nn_layer *l = layers;
  l->input.data = x;
  for (; !NN_LAYER_EMPTY(l); l++) {
    l->forward(l, 0);
  }
  return (l - 1)->output.data;
}

float nn_train(struct nn_layer *layers, float *x, float *y, float rate) {
  unsigned int i;
  float *e;
  float error = 0;
  struct nn_layer *l = layers;
  l->input.data = x;
  for (; !NN_LAYER_EMPTY(l); l++) {
    l->forward(l, 0);
  }
  l--;
  e = l->cache.data + l->cache.len;
  for (i = 0; i < l->output.len; i++) {
    e[i] = y[i] - l->output.data[i];
    error = error + (e[i] * e[i]) / l->output.len;
  }
  do {
    l->backward(l, e, rate);
    e = l->errors.data;
  } while (l-- != layers);
  return error;
}

#define NN_ACT_LINEAR(x) (x)
#define NN_ACT_RELU(x) ((x) * ((x) > 0))
#define NN_ACT_LRELU(x) ((x) > 0 ? (x) : 0.01 * (x))
#define NN_ACT_SIGMOID(x) (1.0 / (1.0 + exp(-(x))))
#define NN_ACT_SOFTPLUS(x) log(1.0 + exp(x))

#define NN_ACT_LOOP(i, vec, fn)                                                \
  for ((i) = 0; (i) < (vec).len; (i)++) {                                      \
    (vec).data[(i)] = fn((vec).data[(i)]);                                     \
  }

static void nn_dense_forward(struct nn_layer *l, int training) {
  unsigned int i, j;
  unsigned int n = l->input.len + 1;
  (void)training;
  /* Sum the inputs multiplied by weights */
  for (i = 0; i < l->output.len; i++) {
    float sum = 0;
    /* This loop is likely to be vectorized */
    for (j = 0; j < l->input.len; j++) {
      sum = sum + l->input.data[j] * l->weights.data[i * n + j];
    }
    if ((l->flags & NN_FLAG_NO_BIAS) == 0) {
      sum = sum + l->weights.data[i * n + n - 1];
    }
    l->output.data[i] = sum;
  }
  /* Apply activation function, RELU and LRELU are likely to get vectorized,
   * LINEAR does nothing */
  switch (l->flags & NN_ACTFN) {
  case NN_FLAG_RELU:
    NN_ACT_LOOP(i, l->output, NN_ACT_RELU);
    break;
  case NN_FLAG_LRELU:
    NN_ACT_LOOP(i, l->output, NN_ACT_LRELU);
    break;
  case NN_FLAG_SIGMOID:
    NN_ACT_LOOP(i, l->output, NN_ACT_SIGMOID);
    break;
  case NN_FLAG_SOFTPLUS:
    NN_ACT_LOOP(i, l->output, NN_ACT_SOFTPLUS);
    break;
  }
}

#define NN_DACT_LINEAR(x) 1.0
#define NN_DACT_RELU(x) (1.0 * ((x) > 0))
#define NN_DACT_LRELU(x) ((x) > 0 ? 1.0 : 0.01)
#define NN_DACT_SIGMOID(x) ((x) * (1 - (x)))
#define NN_DACT_SOFTPLUS(x) (1.0 / (1.0 + exp(-x)))

static void nn_dense_backward(struct nn_layer *l, float *e, float rate) {
  unsigned int i, j;
  unsigned int n = l->input.len + 1;
  for (j = 0; j < l->input.len; j++) {
    float sum_e = 0;
    /* These loops are likely to be vectorized */
    switch (l->flags & NN_ACTFN) {
    case NN_FLAG_LINEAR:
      for (i = 0; i < l->output.len; i++) {
	sum_e = sum_e + e[i] * NN_DACT_LINEAR(l->output.data[i]) *
			    l->weights.data[i * n + j];
      }
      break;
    case NN_FLAG_RELU:
      for (i = 0; i < l->output.len; i++) {
	sum_e = sum_e + e[i] * NN_DACT_RELU(l->output.data[i]) *
			    l->weights.data[i * n + j];
      }
      break;
    case NN_FLAG_LRELU:
      for (i = 0; i < l->output.len; i++) {
	sum_e = sum_e + e[i] * NN_DACT_LRELU(l->output.data[i]) *
			    l->weights.data[i * n + j];
      }
      break;
    case NN_FLAG_SIGMOID:
      for (i = 0; i < l->output.len; i++) {
	sum_e = sum_e + e[i] * NN_DACT_SIGMOID(l->output.data[i]) *
			    l->weights.data[i * n + j];
      }
      break;
    case NN_FLAG_SOFTPLUS:
      for (i = 0; i < l->output.len; i++) {
	sum_e = sum_e + e[i] * NN_DACT_SOFTPLUS(l->output.data[i]) *
			    l->weights.data[i * n + j];
      }
      break;
    }
    l->errors.data[j] = sum_e;
  }

  for (i = 0; i < l->output.len; i++) {
    float dsigm = 0;
    switch (l->flags & NN_ACTFN) {
    case NN_FLAG_LINEAR:
      dsigm = NN_DACT_LINEAR(l->output.data[i]);
      break;
    case NN_FLAG_RELU:
      dsigm = NN_DACT_RELU(l->output.data[i]);
      break;
    case NN_FLAG_LRELU:
      dsigm = NN_DACT_LRELU(l->output.data[i]);
      break;
    case NN_FLAG_SIGMOID:
      dsigm = NN_DACT_SIGMOID(l->output.data[i]);
      break;
    case NN_FLAG_SOFTPLUS:
      dsigm = NN_DACT_SOFTPLUS(l->output.data[i]);
      break;
    }
    /* This loop is likely to be vectorized */
    for (j = 0; j < l->input.len; j++) {
      l->weights.data[i * n + j] += rate * e[i] * dsigm * l->input.data[j];
    }
    if ((l->flags & NN_FLAG_NO_BIAS) == 0) {
      l->weights.data[i * n + n - 1] += rate * e[i] * dsigm;
    }
  }
}

#endif /* NN_H */
