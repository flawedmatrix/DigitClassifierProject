#pragma once

class MultiLayerNeuralNetwork : public NeuralNetwork
{
    CuMatrix<float> weights;
    CuMatrix<float> bias;
    errorMeasure fError;
    void initialize(float learningRate, errorMeasure e, int d, int* dims);

public:
    MultiLayerNeuralNetwork(void);
    MultiLayerNeuralNetwork(float learningRate);
    MultiLayerNeuralNetwork(errorMeasure e);
    MultiLayerNeuralNetwork(float learningRate, errorMeasure e);
    MultiLayerNeuralNetwork(float learningRate, errorMeasure e, int d, int* dims);
    ~MultiLayerNeuralNetwork(void);

    void predict(CuMatrix<float> &input, CuMatrix<char> &output);
    void forwardPropagate(CuMatrix<float> &input, CuMatrix<float> &output);
    void sigmoidForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &output);
    void runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels);
};