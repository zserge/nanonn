import unittest
from nn import NN, Dense, sigmoid


class TestNN(unittest.TestCase):
    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid[0](0), 0.5)
        self.assertAlmostEqual(sigmoid[0](2), 0.88079708)
        self.assertAlmostEqual(sigmoid[1](0), 0)
        self.assertAlmostEqual(sigmoid[1](2), -2)

    def test_forward(self):
        l = Dense(
            units=1, inputs=3, weights=[1.74481176, -0.7612069, 0.3190391, -0.24937038]
        )
        z1 = l(None, [1.62434536, -0.52817175, 0.86540763])
        self.assertAlmostEqual(z1[0], 0.96313579)
        z2 = l(None, [-0.61175641, -1.07296862, -2.3015387])
        self.assertAlmostEqual(z1[0], 0.22542973)

    def test_weights(self):
        nn = NN(
            Dense(units=2, inputs=2, weights=[0.15, 0.2, 0.35, 0.25, 0.3, 0.35]),
            Dense(units=2, inputs=2, weights=[0.4, 0.45, 0.6, 0.5, 0.55, 0.6]),
        )
        z = nn.predict([0.05, 0.1])
        self.assertAlmostEqual(z[0], 0.75136507)
        self.assertAlmostEqual(z[1], 0.772928465)
        e = nn.train([0.05, 0.1], [0.01, 0.99], 0)
        self.assertAlmostEqual(e, 0.298371109)
        nn.train([0.05, 0.1], [0.01, 0.99], 0.5)
        print(nn[0].get_weights())
        print(nn[1].get_weights())
        for i, w in enumerate([0.35891, 0.40866, 0.53075, 0.5113, 0.5613, 0.61904]):
            self.assertAlmostEqual(nn[1].get_weights()[i], w, delta=0.01)
        for i, w in enumerate([0.14978, 0.19956, 0.34561, 0.24975, 0.2995, 0.34502]):
            self.assertAlmostEqual(nn[0].get_weights()[i], w, delta=0.01)


if __name__ == "__main__":
    unittest.main()
