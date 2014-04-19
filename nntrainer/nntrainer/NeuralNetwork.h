#pragma once
class NeuralNetwork
{
    int n;
    // A 768 x n matrix of features for all n training points
    CuMatrix<float> tr_features;
    // A 1 x n matrix of labels for all n training points
    CuMatrix<int> itr_labels;
    // A 10 x n matrix of vectors for all n training points,
    // where each column is a 10x1 vector x = [0 ... 0] except
    // x[i] = 1 if the label is i
    CuMatrix<float> ftr_labels;

    // A 768 x m matrix of features for all m test points
    CuMatrix<float> test_features;
    // A 1 x m matrix of labels for all m test points
    CuMatrix<int> test_labels;

protected:
    NeuralNetwork() {}
    ~NeuralNetwork() {}

public:
    virtual void predict(CuMatrix<float> &input, CuMatrix<float> &output) = 0;
    virtual void backPropagationUpdate() = 0;
    virtual void runTrainingIteration(CuMatrix<float> &data) = 0;


    void runTrainingEpoch();
    void runEpochs(int epochs);
    void loadData();
    float calculateError(CuMatrix<float> &features, CuMatrix<int> &labels);
};

