#pragma once
#include <vector>

#define IDIM_ONE 300
#define IDIM_TWO 100

class MultiLayerNeuralNetwork : public NeuralNetwork
{
    CuMatrix<float> weights_01;
    CuMatrix<float> weights_12;
    CuMatrix<float> weights_23;
    CuMatrix<float> biases_01;
    CuMatrix<float> biases_12;
    CuMatrix<float> biases_23;

    errorMeasure fError;
    void initialize(float learningRate, errorMeasure e);

public:
    MultiLayerNeuralNetwork(void);
    MultiLayerNeuralNetwork(float learningRate);
    MultiLayerNeuralNetwork(errorMeasure e);
    MultiLayerNeuralNetwork(float learningRate, errorMeasure e);
    ~MultiLayerNeuralNetwork(void);

    void transformData(CuMatrix<float> &input);
    void predict(CuMatrix<float> &input, CuMatrix<char> &output);
    void tanhForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &output, size_t d);
    void sigmoidForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &output);
    void runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels);
};