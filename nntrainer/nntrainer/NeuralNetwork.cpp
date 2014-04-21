#include "constants.h"
#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds milliseconds;

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
    tr_features.loadSelection(shuffled_elems);
    ftr_labels.loadSelection(shuffled_elems);

    CuMatrix<float> data(NUM_FEATURES, MINIBATCH_SIZE);
    CuMatrix<float> labels(DIGITS, MINIBATCH_SIZE);
    for (size_t s = 0; s < max_iters; s += MINIBATCH_SIZE) {
        tr_features.selectData(data, s, MINIBATCH_SIZE);
        ftr_labels.selectData(labels, s, MINIBATCH_SIZE);
        runTrainingIteration(data, labels);
    }
}

void NeuralNetwork::runEpochs(unsigned int epochs) {
    // Test initial error
    float train_error = calculateError(tr_features, itr_labels);
    float test_error = calculateError(test_features, test_labels);
    std::cout << "Initial errors:" << std::endl;
    std::cout << "Training error " << train_error << std::endl;
    std::cout << "Test error " << test_error << std::endl;

    for (unsigned int i = 0; i < epochs; i++) {
        Clock::time_point t0 = Clock::now();
        runTrainingEpoch();
        Clock::time_point t1 = Clock::now();
        milliseconds ms = std::chrono::duration_cast<milliseconds>(t1 - t0);
        std::cout << "Training took " << ms.count() << " milliseconds" << std::endl;
        train_error = calculateError(tr_features, itr_labels);
        test_error = calculateError(test_features, test_labels);
        std::cout << "Epoch " << i << " completed" << std::endl;
        std::cout << "Training error " << train_error << std::endl;
        std::cout << "Test error " << test_error << std::endl;
    }
}

size_t readData(const char *filename, size_t divisor, char** out) {
    std::FILE *fp = std::fopen(filename, "rb");
    if (!fp) { fprintf(stderr,"Could not find file %s\n", filename); exit(1); }
    std::fseek(fp, 0, SEEK_END);
    size_t num_bytes = std::ftell(fp);
    std::cout << filename << ", " << num_bytes << " bytes" << std::endl;
    if (num_bytes % divisor > 0) { fprintf(stderr,"%s is in an invalid format!\n", filename); exit(1); }

    char *data = new char[num_bytes];
    std::rewind(fp);
    std::fread(data, 1, num_bytes, fp);
    std::fclose(fp);
    *out = data;
    return num_bytes;
}

void NeuralNetwork::loadData() {
    char *x_train = 0;
    char *y_train = 0;
    char *x_test = 0;
    char *y_test = 0;

    // Load the training data
    size_t train_size = readData("x_train", NUM_FEATURES, &x_train);
    train_size /= NUM_FEATURES;
    readData("y_train", train_size, &y_train);

    CuMatrix<char> itr_features = CuMatrix<char>(NUM_FEATURES, train_size);
    tr_features = CuMatrix<float>(NUM_FEATURES, train_size);
    itr_features.loadDataFrom(x_train);
    itr_features.toFloat(tr_features);
    tr_features.normalize(256);

    itr_labels = CuMatrix<char>(1, train_size);
    ftr_labels = CuMatrix<float>(DIGITS, train_size);
    itr_labels.loadDataFrom(y_train);
    itr_labels.encode(ftr_labels, DIGITS);

    // Load the test data
    size_t test_size = readData("x_test", NUM_FEATURES, &x_test);
    test_size /= NUM_FEATURES;
    readData("y_test", test_size, &y_test);

    CuMatrix<char> itest_features = CuMatrix<char>(NUM_FEATURES, test_size);
    test_features = CuMatrix<float>(NUM_FEATURES, test_size);
    itest_features.loadDataFrom(x_test);
    itest_features.toFloat(test_features);
    test_features.normalize(256);

    test_labels = CuMatrix<char>(1, test_size);
    test_labels.loadDataFrom(y_test);
}

float NeuralNetwork::calculateError(CuMatrix<float> &data, CuMatrix<char> &labels) {
    // Find the forward outputs for the training set
    int n = data.getCols();
    CuMatrix<char> prediction(1, n);
    predict(data, prediction);
    CuMatrix<char> errors(1, n);
    CuMatrix<char>::notEquals(labels, prediction, errors);
    int num_error = errors.reduce();
    return (float)num_error/n;
}