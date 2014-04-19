#pragma once
class NeuralNetwork
{
public:
    NeuralNetwork(void);
    virtual void trainingEpoch() = 0;
    ~NeuralNetwork(void);
};

