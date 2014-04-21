#include "constants.h"
#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include "MultiLayerNeuralNetwork.h"

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(void) {
    initialize(0.1f, MEAN_SQUARED);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(float learningRate) {
    initialize(learningRate, MEAN_SQUARED);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(errorMeasure e) {
    initialize(0.1f, e);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(float learningRate, errorMeasure e) {
    initialize(learningRate, e);
}

void MultiLayerNeuralNetwork::initialize(float alpha, errorMeasure e)  {
    weights_01 = CuMatrix<float>(NUM_FEATURES, IDIM_ONE);
    weights_12 = CuMatrix<float>(IDIM_ONE, IDIM_TWO);
    weights_23 = CuMatrix<float>(IDIM_TWO, DIGITS);
    biases_01 = CuMatrix<float>(IDIM_ONE, 1);
    biases_12 = CuMatrix<float>(IDIM_TWO, 1);
    biases_23 = CuMatrix<float>(DIGITS, 1);

    weights_01.initRandom();
    weights_01.normalize(256);
    weights_12.initRandom();
    weights_12.normalize(256);
    weights_23.initRandom();
    weights_23.normalize(256);
    biases_01.initRandom();
    biases_01.normalize(256);
    biases_12.initRandom();
    biases_12.normalize(256);
    biases_23.initRandom();
    biases_23.normalize(256);

    learningRate = alpha;
    fError = e;
}

MultiLayerNeuralNetwork::~MultiLayerNeuralNetwork(void)
{
}

void MultiLayerNeuralNetwork::transformData(CuMatrix<float> &input) {
    input.standardize();
    input.normalize(16);
};

void MultiLayerNeuralNetwork::predict(CuMatrix<float> &input, CuMatrix<char> &output) {
    size_t n = input.getCols();
    CuMatrix<float> y1 = CuMatrix<float>(IDIM_ONE, n);
    CuMatrix<float> y2 = CuMatrix<float>(IDIM_TWO, n);
    CuMatrix<float> y3 = CuMatrix<float>(DIGITS, n);
    tanhForwardPropagate(input, y1, 0);
    tanhForwardPropagate(y1, y2, 1);
    sigmoidForwardPropagate(y2, y3);
    y3.argmax(output);
}

void MultiLayerNeuralNetwork::tanhForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &y, size_t d) {
    if (d == 0) {
        CuMatrix<float>::multiply(weights_01, true, input, false, y);
        CuMatrix<float>::addVector(y, biases_01, y);
    } else if (d == 1) {
        CuMatrix<float>::multiply(weights_12, true, input, false, y);
        CuMatrix<float>::addVector(y, biases_12, y);
    } else {
        throw "Invalid call to tanhForwardPropagate";
    }
    y.applyTanh();
}

void MultiLayerNeuralNetwork::sigmoidForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &y) {
    CuMatrix<float>::multiply(weights_23, true, input, false, y);
    CuMatrix<float>::addVector(y, biases_23, y);
    y.applySigmoid();
}

void MultiLayerNeuralNetwork::runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels) {
    size_t n = data.getCols();
    // Forward propagation
    CuMatrix<float> y1 = CuMatrix<float>(IDIM_ONE, n);
    CuMatrix<float> y2 = CuMatrix<float>(IDIM_TWO, n);
    CuMatrix<float> y3 = CuMatrix<float>(DIGITS, n);
    tanhForwardPropagate(data, y1, 0);
    tanhForwardPropagate(y1, y2, 1);
    sigmoidForwardPropagate(y2, y3);

    // Back propagation step
    CuMatrix<float> dl3(DIGITS, n);
    CuMatrix<float> dl2(IDIM_TWO, n);
    CuMatrix<float> dl1(IDIM_ONE, n);

    // Calculate delta for last layer
    switch (fError) {
    case MEAN_SQUARED:
    {
        // Calculate (y-t)
        CuMatrix<float> ymt(DIGITS, n);
        CuMatrix<float>::sub(y3, labels, ymt);
        // Calculate y-y^2
        CuMatrix<float>::hadm(y3, y3, dl3);
        CuMatrix<float>::sub(y3, dl3, dl3);
        // Calculate (y-t)*(y-y^2)
        CuMatrix<float>::hadm(ymt, dl3, dl3);
        break;
    }
    case CROSS_ENTROPY:
    {
        // Calculate (y-t)
        CuMatrix<float>::sub(y3, labels, dl3);
        break;
    }
    default:
        throw "UNKNOWN ERROR FUNCTION";
        break;
    }
    // Back propagation on layer 2
    // Calculate (W(2)*d(2))
    CuMatrix<float>::multiply(weights_23, false, dl3, false, dl2);
    // Calculate tanh' = (1-y(2)^2)
    CuMatrix<float> yp2(IDIM_TWO, n);
    CuMatrix<float> one2(IDIM_TWO, n);
    one2.fill(1.0f);
    CuMatrix<float>::hadm(y2, y2, yp2);
    CuMatrix<float>::sub(one2, yp2, yp2);
    // Calculate (1-y(2)^2)*(W(2)*d(2))
    CuMatrix<float>::hadm(yp2, dl2, dl2);

    // Back propagation on layer 1
    // Calculate (W(2)*d(1))
    CuMatrix<float>::multiply(weights_12, false, dl2, false, dl1);
    // Calculate tanh' = (1-y(1)^2)
    CuMatrix<float> yp1(IDIM_ONE, n);
    CuMatrix<float> one1(IDIM_ONE, n);
    one1.fill(1.0f);
    CuMatrix<float>::hadm(y1, y1, yp1);
    CuMatrix<float>::sub(one1, yp1, yp1);
    // Calculate (1-y(1)^2)*(W(1)*d(1))
    CuMatrix<float>::hadm(yp1, dl1, dl1);

    // Update weights and biases

    // Update W(0)
    CuMatrix<float> dW0(NUM_FEATURES, IDIM_ONE);
    CuMatrix<float>::multiply(data, false, dl1, true, dW0);
    dW0.scale(learningRate);
    CuMatrix<float>::sub(weights_01, dW0, weights_01);

    // Update W(1)
    CuMatrix<float> dW1(IDIM_ONE, IDIM_TWO);
    CuMatrix<float>::multiply(y1, false, dl2, true, dW1);
    dW1.scale(learningRate);
    CuMatrix<float>::sub(weights_12, dW1, weights_12);

    // Update W(2)
    CuMatrix<float> dW2(IDIM_TWO, DIGITS);
    CuMatrix<float>::multiply(y2, false, dl3, true, dW2);
    dW2.scale(learningRate);
    CuMatrix<float>::sub(weights_23, dW2, weights_23);

    CuMatrix<float> one(n, 1);
    one.fill(learningRate);

    // Update B(0)
    CuMatrix<float> dB0(IDIM_ONE, 1);
    CuMatrix<float>::multiply(dl1, false, one, false, dB0);
    CuMatrix<float>::sub(biases_01, dB0, biases_01);

    // Update B(1)
    CuMatrix<float> dB1(IDIM_TWO, 1);
    CuMatrix<float>::multiply(dl2, false, one, false, dB1);
    CuMatrix<float>::sub(biases_12, dB1, biases_12);

    // Update B(2)
    CuMatrix<float> dB2(DIGITS, 1);
    CuMatrix<float>::multiply(dl3, false, one, false, dB2);
    CuMatrix<float>::sub(biases_23, dB2, biases_23);
}