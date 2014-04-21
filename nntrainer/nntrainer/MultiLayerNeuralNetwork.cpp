#include "constants.h"
#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include "MultiLayerNeuralNetwork.h"
#include <iostream>

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(void) {
    int[] dims = {784, 300, 100, 10};
    initialize(0.1f, MEAN_SQUARED, 3, dims);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(float learningRate) {
    int[] dims = {784, 300, 100, 10};
    initialize(learningRate, MEAN_SQUARED, 3, dims);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(errorMeasure e) {
    int[] dims = {784, 300, 100, 10};
    initialize(0.1f, e, 3, dims);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(float learningRate, errorMeasure e) {
    int[] dims = {784, 300, 100, 10};
    initialize(learningRate, e, 3, dims);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(float learningRate, errorMessage e, int d, int* dims) {
    initialize(learningRate, e, d, dims);
}

void MultiLayerNeuralNetwork::initialize(float alpha, errorMeasure e, int d, int* dims)  {
    std::vector<CuMatrix> weights(d);
    std::vector<CuMatrix> biases(d);
    for (int i = 0; i < d; i++) {
        weight = CuMatrix<float>(dims[i], dims[i+1]);
        weight.initRandom();
        weights.at(i) = weight;
        bias = CuMatrix<float>(dims[i+1], 1);
        bias.initRandom();
        biases.at(i) = bias;
    }

    learningRate = alpha;
    fError = e;
    depth = d;
    dimensions = dims;
}

MultiLayerNeuralNetwork::~MultiLayerNeuralNetwork(void)
{
}

void MultiLayerNeuralNetwork::predict(CuMatrix<float> &input, CuMatrix<char> &output) {
    y = input;
    for (int l = 1; l < depth; l++) {
        CuMatrix<float> yn(dimensions[l], n)
        forwardPropagate(y, yn);
        y = yn;
    }
    CuMatrix<float> yn(dimensions[depth], n)
    sigmoidForwardPropagate(y, yn);
    yn.argmax(output);
}

void MultiLayerNeuralNetwork::forwardPropagate(CuMatrix<float> &input, CuMatrix<float> &y) {
    CuMatrix<float>::multiply(weights, true, input, false, y);
    CuMatrix<float>::addVector(y, bias, y);
    y.applySigmoid();
}

void MultiLayerNeuralNetwork::sigmoidForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &y) {
    CuMatrix<float>::multiply(weights, true, input, false, y);
    CuMatrix<float>::addVector(y, bias, y);
    y.applySigmoid();
}

void MultiLayerNeuralNetwork::runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels) {
    int n = data.getCols();
    // Forward propagation
    std::vector<CuMatrix> y_vals(depth);
    y_vals.at(0) = data;
    for (int l = 1; l < depth; l++) {
        CuMatrix<float> y(dimensions[l], n);
        forwardPropagate(y_vals.at(l-1), y);
        y_vals.at(l) = y;
    }
    CuMatrix<float> y(dimensions[depth], n);
    sigmoidForwardPropagate(y_vals.at(depth-1), y);

    std::vector<CuMatrix> dls(depth);
    // Calculate delta for last layer
    CuMatrix<float> dl(dimensions[depth], n);
    switch (fError) {
    case MEAN_SQUARED:
    {
        // Calculate (y-t)
        CuMatrix<float> ymt(dimensions[depth], n);
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
    dls.at(depth-1) = dl;

    // Back propagation on rest of layers
    for (int l = depth; l > 1; l--) {
        CuMatrix<float> dln(dimensions[l-1], n);
        // Calculate (W(l)*d(l))
        CuMatrix<float>::multiply(weights.at(l-1), false, dls.at(l-1), false, dln)
        // Calculate (1-y(l-1)^2)
        CuMatrix<float> yn(dimensions[l-1], n);
        CuMatrix<float> one(dimensions[l-1], n);
        CuMatrix<float>::hadm(y_vals.at(l-1), y_vals.at(l-1), yn);
        CuMatrix<float>::sub(one, yn, yn);
        // Calculate (1-y(l-1)^2)*(W(l)*d(l))
        CuMatrix<float>::hadm(yn, dln, dln)
        dls.at(l-2) = dln;
    }

    // Update weights and biases
    for (int l = 0; l < depth; l++) {
        // Update W(l)
        CuMatrix<float> dW(dimensions[l], dimensions[l+1]);
        CuMatrix<float>::multiply(y_vals.at(l), false, dls.at(l), true, dW);
        dW.scale(learningRate);
        CuMatrix<float>::sub(weights.at(l), dW, weights.at(l));

        // Update B(l)
        CuMatrix<float> one(n, 1);
        one.fill(learningRate);
        CuMatrix<float> dB(dimensions[l+1], 1);
        CuMatrix<float>::multiply(dl, false, one, false, dB);
        CuMatrix<float>::sub(biases.at(l), dB, biases.at(l));
    }
}