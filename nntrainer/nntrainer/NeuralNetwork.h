#pragma once
#include <vector>

class NeuralNetwork
{
protected:
    NeuralNetwork() {}
    ~NeuralNetwork() {}
    // A NUM_FEATURES x n matrix of normalized features for all n training points
    CuMatrix<float> tr_features;
    // A 1 x n matrix of labels for all n training points
    CuMatrix<char> itr_labels;
    // A DIGITS x n matrix of vectors for all n training points,
    // where each column is a 10x1 vector x = [0 ... 0] except
    // x[i] = 1 if the label is i
    CuMatrix<float> ftr_labels;

    // A NUM_FEATURES x m matrix of normalized features for all m test points
    CuMatrix<float> test_features;
    // A 1 x m matrix of labels for all m test points
    CuMatrix<char> test_labels;
    
    std::vector<float> trainErrors;
    std::vector<float> testErrors;

    float learningRate;

public:
    virtual void predict(CuMatrix<float> &input, CuMatrix<char> &output) = 0;
    virtual void runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels) = 0;
    virtual void transformData(CuMatrix<float> &data) = 0;

    void runTrainingEpoch();
    void runEpochs(unsigned int epochs);
    void loadData();
    void writeData(std::string filename);
    float calculateError(CuMatrix<float> &features, CuMatrix<char> &labels);
};

enum errorMeasure {
    MEAN_SQUARED, CROSS_ENTROPY
};