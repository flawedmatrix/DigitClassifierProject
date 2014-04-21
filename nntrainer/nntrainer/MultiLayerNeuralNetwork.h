#pragma once

class SingleLayerNeuralNetwork : public NeuralNetwork
{
    CuMatrix<float> weights;
    CuMatrix<float> bias;
    errorMeasure fError;
    void initialize(float learningRate, errorMeasure e, int d, int* dims);

public:
    SingleLayerNeuralNetwork(void);
    SingleLayerNeuralNetwork(float learningRate);
    SingleLayerNeuralNetwork(errorMeasure e);
    SingleLayerNeuralNetwork(float learningRate, errorMeasure e);
    SingleLayerNeuralNetwork(float learningRate, errorMeasure e, int d, int* dims);
    ~SingleLayerNeuralNetwork(void);

    void predict(CuMatrix<float> &input, CuMatrix<char> &output);
    void forwardPropagate(CuMatrix<float> &input, CuMatrix<float> &output);
    void sigmoidForwardPropagate(CuMatrix<float> &input, CuMatrix<float> &output);
    void runTrainingIteration(CuMatrix<float> &data, CuMatrix<float> &labels);
};