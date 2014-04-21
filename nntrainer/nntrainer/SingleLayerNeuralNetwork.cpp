#include "constants.h"
#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include "SingleLayerNeuralNetwork.h"
#include <iostream>

SingleLayerNeuralNetwork::SingleLayerNeuralNetwork(void)
{
    weights = CuMatrix<float>(NUM_FEATURES, DIGITS);
    weights.initRandom();
    bias = CuMatrix<float>(DIGITS, 1);
    bias.initRandom();

    learningRate = 0.1f;
}

SingleLayerNeuralNetwork::~SingleLayerNeuralNetwork(void)
{
}

void SingleLayerNeuralNetwork::predict(CuMatrix<float> &input, CuMatrix<char> &output) {
    CuMatrix<float> y(DIGITS, input.getCols());
    forwardPropagate(input, y);
    y.argmax(output);
}

void SingleLayerNeuralNetwork::forwardPropagate(CuMatrix<float> &input, CuMatrix<float> &y) {
    float *weightData = weights.returnData();
    float *biasData = bias.returnData();
    std::cout << "Weight " << weightData[400] << std::endl;
    std::cout << "Bias ";
    for (int i = 0; i < 10; i++) {
        std::cout << biasData[i];
    }
    std::cout << std::endl;
    CuMatrix<float>::multiply(weights, true, input, false, y);
    CuMatrix<float>::addVector(y, bias, y);
    y.applySigmoid();
}

void SingleLayerNeuralNetwork::runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels) {
    int n = data.getCols();
    // Forward propagation
    CuMatrix<float> y(DIGITS, n);
    forwardPropagate(data, y);

    // Back propagation step
    // Calculate delta for the last layer
    // Calculate (y-t)
    CuMatrix<float> ymt(DIGITS, n);
    CuMatrix<float>::sub(y, labels, ymt);

    CuMatrix<float> dl(DIGITS, n);
    // Calculate y-y^2
    CuMatrix<float>::hadm(y, y, dl);
    CuMatrix<float>::sub(y, dl, dl);
    // Calculate (y-t)*(y-y^2)
    CuMatrix<float>::hadm(ymt, dl, dl);
    
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