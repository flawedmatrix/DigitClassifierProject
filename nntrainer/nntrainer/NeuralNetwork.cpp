#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

#define MINIBATCH_SIZE 200

void NeuralNetwork::runTrainingEpoch() {
    size_t num_cols = tr_features.getCols();
    size_t max_iters = (size_t)ceil((float)num_cols / MINIBATCH_SIZE) * MINIBATCH_SIZE;
    unsigned int *shuffled_elems = new unsigned int[max_iters];

    for (size_t i = 0; i < num_cols; i++) {
        shuffled_elems[i] = i;
    }

    unsigned int seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
    // Shuffle data
    std::shuffle(shuffled_elems, shuffled_elems + num_cols, std::default_random_engine(seed));
    // Provide a little wrap-around
    for (size_t i = num_cols; i < max_iters; i++) {
        shuffled_elems[i] = shuffled_elems[i - num_cols];
    }

    CuMatrix<float> data(768, MINIBATCH_SIZE);
    CuMatrix<float> labels(10, MINIBATCH_SIZE);
    for (size_t s = 0; s < max_iters; s += MINIBATCH_SIZE) {
        tr_features.selectData(data, shuffled_elems + s, MINIBATCH_SIZE);
        ftr_labels.selectData(labels, shuffled_elems + s, MINIBATCH_SIZE);
        runTrainingIteration(data, labels);
    }
}

void NeuralNetwork::runEpochs(unsigned int epochs) {
    for (unsigned int i = 0; i < epochs; i++) {
        runTrainingEpoch();
        float train_error = calculateError(tr_features, itr_labels);
        float test_error = calculateError(test_features, test_labels);
        std::cout << "Epoch " << i << " completed" << std::endl;
        std::cout << "Training error " << train_error << std::endl;
        std::cout << "Test error " << test_error << std::endl;
    }
}

void NeuralNetwork::loadData() {

}

float NeuralNetwork::calculateError(CuMatrix<float> &data, CuMatrix<int> &labels) {
    // Find the forward outputs for the training set
    int n = data.getCols();
    CuMatrix<int> prediction = CuMatrix<int>(1, n);
    predict(data, prediction);
    CuMatrix<int> errors = CuMatrix<int>(1, n);
    CuMatrix<int>::notEquals(labels, prediction, errors);
    int num_error = errors.reduce();
    return (float)num_error/n;
}