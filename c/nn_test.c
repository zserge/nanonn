#include "nn.h"
#include <stdio.h>

int status = 0;

#define ASSERT_EQ(a, b)                                                        \
  do {                                                                         \
    if (fabs((a) - (b)) > 0.001) {                                             \
      printf("FAIL:%d: %f != %f\n", __LINE__, a, b);                           \
      status = 1;                                                              \
    }                                                                          \
  } while (0)

static void test_network_forward() {
  float x[3];
  float *z;
  float mem[64];
  struct nn_layer nn[] = {
      NN_DENSE(3, 1, NN_FLAG_SIGMOID),
      NN_EMPTY(),
  };
  nn_init(nn, mem, sizeof(mem));
  nn[0].weights.data[0] = 1.74481176;
  nn[0].weights.data[1] = -0.7612069;
  nn[0].weights.data[2] = 0.3190391;
  nn[0].weights.data[3] = -0.24937038;
  x[0] = 1.62434536;
  x[1] = -0.52817175;
  x[2] = 0.86540763;
  z = nn_predict(nn, x);
  ASSERT_EQ(z[0], 0.96313579);
  x[0] = -0.61175641;
  x[1] = -1.07296862;
  x[2] = -2.3015387;
  z = nn_predict(nn, x);
  ASSERT_EQ(z[0], 0.22542973);
}

static void test_network_backward() {
  float x[2];
  float y[2];
  float *z;
  float e;
  float mem[1024];
  struct nn_layer nn[] = {
      NN_DENSE(2, 2, NN_FLAG_SIGMOID),
      NN_DENSE(2, 2, NN_FLAG_SIGMOID),
      NN_EMPTY(),
  };
  nn_init(nn, mem, sizeof(mem));
  nn[0].weights.data[0] = 0.15;
  nn[0].weights.data[1] = 0.2;
  nn[0].weights.data[2] = 0.35;
  nn[0].weights.data[3] = 0.25;
  nn[0].weights.data[4] = 0.3;
  nn[0].weights.data[5] = 0.35;

  nn[1].weights.data[0] = 0.4;
  nn[1].weights.data[1] = 0.45;
  nn[1].weights.data[2] = 0.6;
  nn[1].weights.data[3] = 0.5;
  nn[1].weights.data[4] = 0.55;
  nn[1].weights.data[5] = 0.6;

  x[0] = 0.05;
  x[1] = 0.1;
  z = nn_predict(nn, x);
  ASSERT_EQ(z[0], 0.75136507);
  ASSERT_EQ(z[1], 0.772928465);
  y[0] = 0.01;
  y[1] = 0.99;
  e = nn_train(nn, x, y, 0);
  ASSERT_EQ(e, 0.298371109);
  nn_train(nn, x, y, 0.5);
  nn_print(nn);
  ASSERT_EQ(nn[1].weights.data[0], 0.35891);
  ASSERT_EQ(nn[1].weights.data[1], 0.40866);
  ASSERT_EQ(nn[1].weights.data[2], 0.53075);
  ASSERT_EQ(nn[1].weights.data[3], 0.5113);
  ASSERT_EQ(nn[1].weights.data[4], 0.5613);
  ASSERT_EQ(nn[1].weights.data[5], 0.61904);

  ASSERT_EQ(nn[0].weights.data[0], 0.14978);
  ASSERT_EQ(nn[0].weights.data[1], 0.19956);
  ASSERT_EQ(nn[0].weights.data[2], 0.34561);
  ASSERT_EQ(nn[0].weights.data[3], 0.24975);
  ASSERT_EQ(nn[0].weights.data[4], 0.2995);
  ASSERT_EQ(nn[0].weights.data[5], 0.34502);
}

int main() {
  test_network_forward();
  test_network_backward();
  return status;
}
