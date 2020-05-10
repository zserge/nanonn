package com.github.zserge.nanonn

import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Test

class NanoNNTest {
    @Test
    fun testDenseLayer() {
        assertEquals(4, 2 * 2)
    }

    @Test
    fun testSigmoid() {
        var y = Sigmoid.act(0f)
        assertEquals(0.5f, y)

        y = Sigmoid.act(2f)
        assertEquals(0.88079708, y.toDouble(), 0.0001)
    }

    @Test
    fun testForward() {
        val l = Dense(3, 1, Sigmoid)
        l.weights = floatArrayOf(1.74481176f, -0.7612069f, 0.3190391f, -0.24937038f)

        val x1 = floatArrayOf(1.62434536f, -0.52817175f, 0.86540763f)
        val y1 = 0.96313579f
        val z1 = l.forward(x1)
        assertEquals(y1.toDouble(), z1[0].toDouble(), 0.001)

        val x2 = floatArrayOf(-0.61175641f, -1.07296862f, -2.3015387f)
        val y2 = 0.22542973f
        val z2 = l.forward(x2)
        assertEquals(y2.toDouble(), z2[0].toDouble(), 0.001)
    }

    /**
     * @see <a href="https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/">backpropagation example</a>
     */
    @Test
    fun testWeights() {
        val l1 = Dense(2, 2)
        val l2 = Dense(2, 2)
        val n = Network(l1, l2)

        // Initialize weights
        l1.weights = floatArrayOf(0.15f, 0.2f, 0.35f, 0.25f, 0.3f, 0.35f)
        l2.weights = floatArrayOf(0.4f, 0.45f, 0.6f, 0.5f, 0.55f, 0.6f)

        // Ensure forward propagation works for both layers
        val z = n.predict(floatArrayOf(0.05f, 0.1f))
        assertEquals(0.75136507f, z[0], 0.0001f)
        assertEquals(0.772928465f, z[1], 0.0001f)

        // Ensure that squared error is calculated correctly (use rate=0 to avoid training)
        val e = n.train(floatArrayOf(0.05f, 0.1f), floatArrayOf(0.01f, 0.99f), 0f)
        assertEquals(0.298371109f, e, 0.0001f)

        // Backpropagation with rate 0.5
        n.train(floatArrayOf(0.05f, 0.1f), floatArrayOf(0.01f, 0.99f), 0.5f)

        // Check weights
        floatArrayOf(
                0.35891648f,
                0.408666186f,
                0.530751f,
                0.511301270f,
                0.561370121f,
                0.619049f
        ).forEachIndexed { i, w ->
            assertEquals(w, l2.weights[i], 0.001f)
        }
        floatArrayOf(
                0.149780716f,
                0.19956143f,
                0.345614f,
                0.24975114f,
                0.29950229f,
                0.345023f
        ).forEachIndexed { i, w ->
            assertEquals(w, l1.weights[i], 0.001f)
        }
    }

    // Use hidden layer to predict XOR function
    @Test
    fun testNetworkXor() {
        val x = arrayOf(
                floatArrayOf(0f, 0f),
                floatArrayOf(0f, 1f),
                floatArrayOf(1f, 0f),
                floatArrayOf(1f, 1f)
        )
        val y = arrayOf(
                floatArrayOf(0f),
                floatArrayOf(1f),
                floatArrayOf(1f),
                floatArrayOf(0f)
        )
        val n = Network(Dense(2, 4), Dense(4, 1))

        // Train for several epochs, or until the error is less than 2%
        repeat((0..10000).count()) {
            var e = 0f
            (x.indices).forEach { i ->
                e += n.train(x[i], y[i], 1f)
            }
            if (e < 0.02f) {
                return
            }
        }
        fail("Failed to train the model")
    }
}
