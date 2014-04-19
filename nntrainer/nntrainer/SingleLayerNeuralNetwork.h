#pragma once
class SingleLayerNeuralNetwork : public NeuralNetwork
{
    CuMatrix<float> weights;
    CuMatrix<float> bias;

public:
    SingleLayerNeuralNetwork(void);
    ~SingleLayerNeuralNetwork(void);

    void predict(CuMatrix<float> &input, CuMatrix<int> &output);
    void forwardPropagate(CuMatrix<float> &input, CuMatrix<float> &output);
    void runTrainingIteration(CuMatrix<float> &data);
};

