#include "constants.h"
#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include <iostream>
#include <deque>
#include "MultiLayerNeuralNetwork.h"

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(void) {
    size_t arr[] = {784, 300, 100, 10};
    std::vector<size_t> dims(arr, arr + sizeof(arr) / sizeof(size_t));
    initialize(0.1f, MEAN_SQUARED, 3, dims);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(float learningRate) {
    size_t arr[] = {784, 300, 100, 10};
    std::vector<size_t> dims(arr, arr + sizeof(arr) / sizeof(size_t));
    initialize(learningRate, MEAN_SQUARED, 3, dims);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(errorMeasure e) {
    size_t arr[] = {784, 300, 100, 10};
    std::vector<size_t> dims(arr, arr + sizeof(arr) / sizeof(size_t));
    initialize(0.1f, e, 3, dims);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(float learningRate, errorMeasure e) {
    size_t arr[] = {784, 300, 100, 10};
    std::vector<size_t> dims(arr, arr + sizeof(arr) / sizeof(size_t));
    initialize(learningRate, e, 3, dims);
}

MultiLayerNeuralNetwork::MultiLayerNeuralNetwork(float learningRate, errorMeasure e, size_t d, std::vector<size_t> dims) {
    initialize(learningRate, e, d, dims);
}

void MultiLayerNeuralNetwork::initialize(float alpha, errorMeasure e, size_t d, std::vector<size_t> dims)  {
    CuMatrix<float> weight;
    CuMatrix<float> bias;
    for (size_t i = 0; i < d; i++) {
        weight = CuMatrix<float>(dims[i], dims[i+1]);
        weight.initRandom();
        weights.push_back(weight);
        bias = CuMatrix<float>(dims[i+1], 1);
        bias.initRandom();
        biases.push_back(bias);
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
    CuMatrix<float> y = CuMatrix<float>(input);
    CuMatrix<float> yn;
    size_t n = y.getCols();
    for (size_t l = 1; l < depth; l++) {
        yn = CuMatrix<float>(dimensions[l], n);
        tanhForwardPropagate(y, yn, l - 1);
        y = yn;
    }
    yn = CuMatrix<float>(dimensions[depth], n);
    sigmoidForwardPropagate(y, yn, depth - 1);
    yn.argmax(output);
}

void MultiLayerNeuralNetwork::tanhForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &y, size_t d) {
    CuMatrix<float>::multiply(weights[d], true, input, false, y);
    CuMatrix<float>::addVector(y, biases[d], y);
    y.applyTanh();
}

void MultiLayerNeuralNetwork::sigmoidForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &y, size_t d) {
    CuMatrix<float>::multiply(weights[d], true, input, false, y);
    CuMatrix<float>::addVector(y, biases[d], y);
    y.applySigmoid();
}

void MultiLayerNeuralNetwork::runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels) {
    size_t n = data.getCols();
    // Forward propagation
    std::vector<CuMatrix<float> > y_vals;
    y_vals.push_back(CuMatrix<float>(data));
    CuMatrix<float> y;
    for (size_t l = 1; l < depth; l++) {
        y = CuMatrix<float>(dimensions[l], n);
        tanhForwardPropagate(y_vals[l-1], y, l-1);
        y_vals.push_back(y);
    }
    y = CuMatrix<float>(dimensions[depth], n);
    sigmoidForwardPropagate(y_vals[depth-1], y, depth-1);

    // Back propagation step
    std::deque<CuMatrix<float> > dls;

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
    dls.push_front(dl);

    // Back propagation on rest of layers
    for (size_t l = depth-1; l > 0; l--) {
        CuMatrix<float> dln(dimensions[l], n);
        // Calculate (W(l)*d(l))
        CuMatrix<float>::multiply(weights[l], false, dls.front(), false, dln);
        // Calculate tanh' = (1-y(l-1)^2)
        CuMatrix<float> yn(dimensions[l], n);
        CuMatrix<float> one(dimensions[l], n);
        one.fill(1.0f);
        CuMatrix<float>::hadm(y_vals[l], y_vals[l], yn);
        CuMatrix<float>::sub(one, yn, yn);
        // Calculate (1-y(l-1)^2)*(W(l)*d(l))
        CuMatrix<float>::hadm(yn, dln, dln);
        dls.push_front(dln);
    }

    // Update weights and biases
    for (size_t l = 0; l < depth; l++) {
        // Update W(l)
        CuMatrix<float> dW(dimensions[l], dimensions[l+1]);
        CuMatrix<float>::multiply(y_vals[l], false, dls[l], true, dW);
        dW.scale(learningRate);
        CuMatrix<float>::sub(weights[l], dW, weights[l]);

        // Update B(l)
        CuMatrix<float> one(n, 1);
        one.fill(learningRate);
        CuMatrix<float> dB(dimensions[l+1], 1);
        CuMatrix<float>::multiply(dls[l], false, one, false, dB);
        CuMatrix<float>::sub(biases[l], dB, biases[l]);
    }
}