#include "constants.h"
#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include "SingleLayerNeuralNetwork.h"
#include <iostream>

SingleLayerNeuralNetwork::SingleLayerNeuralNetwork(void)
{
    initialize(0.1f, MEAN_SQUARED);
}

SingleLayerNeuralNetwork::SingleLayerNeuralNetwork(float learningRate) {
    initialize(learningRate, MEAN_SQUARED);
}

SingleLayerNeuralNetwork::SingleLayerNeuralNetwork(errorMeasure e) {
    initialize(0.1f, e);
}

SingleLayerNeuralNetwork::SingleLayerNeuralNetwork(float learningRate, errorMeasure e) {
    initialize(learningRate, e);
}

void SingleLayerNeuralNetwork::initialize(float alpha, errorMeasure e)  {
    weights = CuMatrix<float>(NUM_FEATURES, DIGITS);
    weights.initRandom();
    bias = CuMatrix<float>(DIGITS, 1);
    bias.initRandom();

    learningRate = alpha;
    fError = e;
}

SingleLayerNeuralNetwork::~SingleLayerNeuralNetwork(void)
{
}

void SingleLayerNeuralNetwork::transformData(CuMatrix<float> &input) {
    input.normalize(1024);
};

void SingleLayerNeuralNetwork::predict(CuMatrix<float> &input, CuMatrix<char> &output) {
    CuMatrix<float> y(DIGITS, input.getCols());
    forwardPropagate(input, y);
    y.argmax(output);
}

void SingleLayerNeuralNetwork::forwardPropagate(CuMatrix<float> &input, CuMatrix<float> &y) {
    CuMatrix<float>::multiply(weights, true, input, false, y);
    CuMatrix<float>::addVector(y, bias, y);
    y.applySigmoid();
}

void SingleLayerNeuralNetwork::runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels) {
    size_t n = data.getCols();
    // Forward propagation
    CuMatrix<float> y(DIGITS, n);
    forwardPropagate(data, y);

    // Back propagation step
    // Calculate delta for the last layer

    CuMatrix<float> dl(DIGITS, n);
    switch (fError) {
    case MEAN_SQUARED:
    {
        // Calculate (y-t)
        CuMatrix<float> ymt(DIGITS, n);
        CuMatrix<float>::sub(y, labels, ymt);
        // Calculate y-y^2
        CuMatrix<float>::hadm(y, y, dl);
        CuMatrix<float>::sub(y, dl, dl);
        // Calculate (y-t)*(y-y^2)
        CuMatrix<float>::hadm(ymt, dl, dl);
        break;
    }
    case CROSS_ENTROPY:
    {
        // Calculate (y-t)
        CuMatrix<float>::sub(y, labels, dl);
        break;
    }
    default:
        throw "UNKNOWN ERROR FUNCTION";
        break;
    }
    
    CuMatrix<float> dW(NUM_FEATURES, DIGITS);
    CuMatrix<float>::multiply(data, false, dl, true, dW);
    dW.scale(learningRate);

    CuMatrix<float> one(n, 1);
    one.fill(learningRate);
    CuMatrix<float> dB(DIGITS, 1);
    CuMatrix<float>::multiply(dl, false, one, false, dB);

    // Update the weight
    CuMatrix<float>::sub(weights, dW, weights);
    // Update the bias
    CuMatrix<float>::sub(bias, dB, bias);
}