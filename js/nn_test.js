import {NN, Dense, sigm} from './nn.js';

function eq(a, b) {
  if (Math.abs(a - b) > 0.0001) {
    throw new Error('Expected ' + a + ' to be equal ' + b);
  }
}

function testSigmoid() {
  console.log('TEST: Sigmoid activation function');
  eq(sigm.f(0), 0.5);
  eq(sigm.f(2), 0.88079708);
}

function testForward() {
  console.log('TEST: Single layer forward propagation');
  const l = Dense({
    units: 1,
    inputs: 3,
    weights: [1.74481176, -0.7612069, 0.3190391, -0.24937038],
  });
  const z1 = l.forward([1.62434536, -0.52817175, 0.86540763]);
  const y1 = 0.96313579;
  eq(z1[0], y1);
  const z2 = l.forward([-0.61175641, -1.07296862, -2.3015387]);
  const y2 = 0.22542973;
  eq(z2[0], y2);
}

function testWeights() {
  console.log('TEST: Two layers backpropagation');
  const l1 = Dense({
    units: 2,
    inputs: 2,
    weights: [0.15, 0.2, 0.35, 0.25, 0.3, 0.35],
  });
  const l2 = Dense({
    units: 2,
    inputs: 2,
    weights: [0.4, 0.45, 0.6, 0.5, 0.55, 0.6],
  });
  const nn = NN(l1, l2);

  const z = nn.predict([0.05, 0.1]);
  eq(z[0], 0.75136507);
  eq(z[1], 0.772928465);

  const e = nn.train([0.05, 0.1], [0.01, 0.99], 0);
  eq(e, 0.298371109);

  nn.train([0.05, 0.1], [0.01, 0.99], 0.5);
  [0.35891, 0.40866, 0.53075, 0.5113, 0.5613, 0.61904].forEach((w, i) =>
    eq(w, l2[i]),
  );
  [0.14978, 0.19956, 0.34561, 0.24975, 0.2995, 0.34502].forEach((w, i) =>
    eq(w, l1[i]),
  );
}

testSigmoid();
testForward();
testWeights();
