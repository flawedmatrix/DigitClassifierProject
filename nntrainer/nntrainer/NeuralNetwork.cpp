#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include <iostream>
#define MINIBATCH_SIZE 200

void NeuralNetwork::runTrainingEpoch() {
    // Shuffle data
    // For each block of 200 training points
    // runTrainingIteration(CuMatrix<float> &data);
}

void NeuralNetwork::runEpochs(int epochs) {
    for (int i = 0; i < epochs; i++) {
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

float NeuralNetwork::calculateError(CuMatrix<float> &features, CuMatrix<int> &labels) {
    // Find the forward outputs for the training set
    CuMatrix<float> prediction = CuMatrix<float>(10, features.getCols());
    predict(features, prediction);
    return 0.0;
}