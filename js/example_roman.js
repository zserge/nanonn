//
// Teaching NN how to add roman numerals.
//

import {NN, Dense} from './nn.js';

//
// Building a dataset. Each number is encoded as 7 positional flags, for each
// possible roman "letter". We only consider numbers 1..10, so the sum will be
// 2..20 (Romans did not have a concept of "zero").
//
// Data set will have 81 samples. Input vector will have 14 elements (two
// numbers), output vector will have 7 elements. Of course, there will be
// overfitting.
//
const roman = [
  // X  I  X  V  I  I  I
  [0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 1, 1],
  [0, 0, 0, 0, 1, 1, 1],
  [0, 1, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 1, 0, 0],
  [0, 0, 0, 1, 1, 1, 0],
  [0, 0, 0, 1, 1, 1, 1],
  [0, 1, 1, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 0],
  [1, 0, 0, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 1, 1],
  [1, 0, 0, 0, 1, 1, 1],
  [1, 1, 0, 1, 0, 0, 0],
  [1, 0, 0, 1, 0, 0, 0],
  [1, 0, 0, 1, 1, 0, 0],
  [1, 0, 0, 1, 1, 1, 0],
  [1, 0, 0, 1, 1, 1, 1],
  [1, 1, 1, 0, 0, 0, 0],
];
const roman2str = y => {
  const digits = ['X', 'I', 'X', 'V', 'I', 'I', 'I'];
  let s = '';
  for (let j = 0; j < 7; j++) {
    if (Math.round(y[j])) {
      s = s + digits[j];
    }
  }
  return s;
};
const X = []; // input vectors
const Y = []; // output vectors
const L = []; // human-readable labels
for (let a = 1; a < 10; a++) {
  for (let b = 1; b < 10; b++) {
    X.push([].concat(roman[a], roman[b]));
    Y.push(roman[a + b]);
    L.push([roman2str(roman[a]), roman2str(roman[b]), roman2str(roman[a + b])]);
  }
}

// Create a small network with two hidden layers
const nn = NN(
  Dense({units: 128, inputs: 14}),
  Dense({units: 20, inputs: 128}),
  Dense({units: 7, inputs: 20}),
);

const evaluate = () => {
  let failed = 0;
  for (let i = 0; i < X.length; i++) {
    const y = nn.predict(X[i]);
    if (roman2str(y) != L[i][2]) {
      console.log(
        '  FAIL:',
        L[i][0],
        '+',
        L[i][1],
        '=',
        L[i][2],
        'not',
        roman2str(y),
      );
      failed++;
    }
  }
  console.log('failed:', failed);
  return failed;
};

const M = 100;
for (let i = 0; i < M; i++) {
  let e = 0;
  for (let j = 0; j < 100; j++) {
    X.forEach((x, k) => {
      e = e + nn.train(x, Y[k], ((M - i) / M) * 0.7 + 0.2);
    });
  }
  console.log('error', e);
  if (evaluate() === 0) {
    break;
  }
}
