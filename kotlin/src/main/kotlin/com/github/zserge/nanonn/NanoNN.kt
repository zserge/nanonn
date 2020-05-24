package com.github.zserge.nanonn

import java.util.*
import kotlin.math.exp
import kotlin.math.ln

/**
 * Network is a sequence of layers
 * <p>
 *     @param layers sequence of layers
 *     @param inputs cache of input vectors passed between the layers
 *     @param errors vector for the input layer
 */
class Network(private vararg val layers: Layer) {

    private val errors: FloatArray
    private val inputs: Array<FloatArray>

    init {
        val n = layers.last().outputsCount
        errors = FloatArray(n)
        inputs = Array(layers.size) { FloatArray(layers.size) }
    }

    /**
     * Passes input vector into the network.
     * @param x input vector
     * @return output vector
     */
    fun predict(x: FloatArray): FloatArray {
        var inputs = x
        layers.forEach {
            val outputs = it.forward(inputs)
            inputs = outputs
        }
        return inputs
    }

    /**
     * Runs one backpropagation iteration through the network.
     * @param x input vector
     * @param y expected output vector
     * @param rate learning rate
     * @return training error
     */
    fun train(x: FloatArray, y: FloatArray, rate: Float): Float {
        var lastInputs = x
        layers.forEachIndexed { i, layer ->
            inputs[i] = lastInputs
            val outputs = layer.forward(lastInputs)
            lastInputs = outputs
        }
        var e = 0f
        y.indices.forEach { i ->
            errors[i] = y[i] - lastInputs[i]
            e += errors[i] * errors[i]
        }
        var errors = errors
        layers.indices.reversed().forEach { i ->
            errors = layers[i].backward(inputs[i], errors, rate)
        }
        return e / y.size.toFloat()
    }
}

/**
 * Layer is a singe network layer. It is an interface, so various layer types
 * can be implemented layer. Layer must return a number of its inputs and
 * outputs. Also it must implement two methods - forward and backpropagation.
 */
interface Layer {
    val inputsCount: Int
    val outputsCount: Int

    fun forward(x: FloatArray): FloatArray
    fun backward(x: FloatArray, e: FloatArray, rate: Float): FloatArray
}

/**
 * Default implementation of a dense fully-connected layer with sigmoid activation function
 * and the given number of inputs and neurons.
 */
class Dense(
        override val inputsCount: Int,
        override val outputsCount: Int,
        private val actFn: ActivationFunction = Sigmoid
) : Layer {

    private val outputs: FloatArray = FloatArray(outputsCount)
    private val errors: FloatArray = FloatArray(inputsCount)
    var weights: FloatArray = FloatArray(outputsCount * (inputsCount + 1))

    private val rand = Random()

    init {
        // Initialize weights randomly
        weights.indices.forEach { i ->
            weights[i] = rand.nextFloat() * 2 - 1
        }
    }

    override fun forward(x: FloatArray): FloatArray {
        val n = inputsCount + 1
        (0 until outputsCount).forEach { i ->
            var sum = 0f
            (0 until inputsCount).forEach { j ->
                sum += x[j] * weights[i * n + j]
            }
            outputs[i] = actFn.act(sum + weights[i * n + n - 1])
        }
        return outputs
    }

    override fun backward(x: FloatArray, e: FloatArray, rate: Float): FloatArray {
        val n = inputsCount + 1
        (0 until inputsCount).forEach { j ->
            errors[j] = 0f
            (0 until outputsCount).forEach { i ->
                errors[j] += e[i] * actFn.dact(outputs[i]) * weights[i * n + j]
            }
        }
        (0 until outputsCount).forEach { i ->
            (0 until inputsCount).forEach { j ->
                weights[i * n + j] += rate * e[i] * actFn.dact(outputs[i]) * x[j]
            }
            weights[i * n + n - 1] += rate * e[i] * actFn.dact(outputs[i])
        }
        return errors
    }
}

/**
 * ActivationFunction is an interface incorporating various activation functions
 * and their derivations.
 */
interface ActivationFunction {
    fun act(x: Float): Float
    fun dact(x: Float): Float
}

object Sigmoid : ActivationFunction {
    override fun act(x: Float) = 1 / (1 + exp(-x))
    override fun dact(x: Float) = x * (1 - x)
}

object SoftPlus : ActivationFunction {
    override fun act(x: Float) = ln(1 + exp(-x))
    override fun dact(x: Float) = 1 / (1 + exp(-x))
}

object Linear : ActivationFunction {
    override fun act(x: Float) = x
    override fun dact(x: Float) = 1f
}

object ReLU : ActivationFunction {
    override fun act(x: Float) = if (x > 0) x else 0f
    override fun dact(x: Float) = if (x > 0) 1f else 0f
}

object LeakyReLU : ActivationFunction {
    override fun act(x: Float) = if (x > 0) x else x * 0.01f
    override fun dact(x: Float) = if (x > 0) 1f else 0.01f
}

