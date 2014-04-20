#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include "SingleLayerNeuralNetwork.h"

SingleLayerNeuralNetwork::SingleLayerNeuralNetwork(void)
{
    weights = CuMatrix<float>(768, 10);
    weights.initRandom();
    bias = CuMatrix<float>(10, 1);
    bias.initRandom();

    learningRate = 0.1f;
}

SingleLayerNeuralNetwork::~SingleLayerNeuralNetwork(void)
{
}

void SingleLayerNeuralNetwork::predict(CuMatrix<float> &input, CuMatrix<int> &output) {
    CuMatrix<float> y(10, input.getCols());
    forwardPropagate(input, y);
    y.argmax(output);
}

void SingleLayerNeuralNetwork::forwardPropagate(CuMatrix<float> &input, CuMatrix<float> &y) {
    CuMatrix<float>::multiply(weights, true, tr_features, false, y);
    CuMatrix<float>::addVector(y, bias, y);
    y.applySigmoid();
}

void SingleLayerNeuralNetwork::runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels) {
    int n = data.getCols();
    // Forward propagation
    CuMatrix<float> y(10, n);
    forwardPropagate(data, y);

    // Back propagation step
    // Calculate delta for the last layer
    CuMatrix<float> ymt(10, n);
    CuMatrix<float>::sub(y, labels, ymt);

    CuMatrix<float> dl(10, n);
    CuMatrix<float>::hadm(y, y, dl);
    CuMatrix<float>::sub(y, dl, dl);
    CuMatrix<float>::hadm(ymt, dl, dl);
    
    CuMatrix<float> dW(768, 10);
    CuMatrix<float>::multiply(tr_features, false, dl, true, dW);
    dW.scale(learningRate);
    dl.scale(learningRate);
    // Update the weight
    CuMatrix<float>::sub(weights, dW, weights);
    // Update the bias
    CuMatrix<float>::sub(bias, dl, bias);
}