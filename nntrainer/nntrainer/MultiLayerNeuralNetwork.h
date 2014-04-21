#pragma once
#include <vector>

class MultiLayerNeuralNetwork : public NeuralNetwork
{
    std::vector<CuMatrix<float> > weights;
    std::vector<CuMatrix<float> > biases;
    errorMeasure fError;
    size_t depth;
    std::vector<size_t> dimensions;
    void initialize(float learningRate, errorMeasure e, size_t d, std::vector<size_t> dims);

public:
    MultiLayerNeuralNetwork(void);
    MultiLayerNeuralNetwork(float learningRate);
    MultiLayerNeuralNetwork(errorMeasure e);
    MultiLayerNeuralNetwork(float learningRate, errorMeasure e);
    MultiLayerNeuralNetwork(float learningRate, errorMeasure e, size_t d, std::vector<size_t> dims);
    ~MultiLayerNeuralNetwork(void);

    void predict(CuMatrix<float> &input, CuMatrix<char> &output);
    void tanhForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &output, size_t d);
    void sigmoidForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &output, size_t d);
    void runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels);
};