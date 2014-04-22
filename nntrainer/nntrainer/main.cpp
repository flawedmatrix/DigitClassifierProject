#include "constants.h"
#include "helpers.cuh"
#include <iostream>
#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include "SingleLayerNeuralNetwork.h"
#include "MultiLayerNeuralNetwork.h"

void run() {
    std::cout << "Starting Single Layer Neural Network with Mean Squared ... " << std::endl;
    SingleLayerNeuralNetwork slnnms(0.1f, MEAN_SQUARED);
    slnnms.loadData();
    slnnms.runEpochs(200);

    std::cout << "Starting Single Layer Neural Network with Cross Entropy ... " << std::endl;
    SingleLayerNeuralNetwork slnnce(0.01f, CROSS_ENTROPY);
    slnnce.loadData();
    slnnce.runEpochs(200);

    std::cout << "Starting Multi Layer Neural Network with Mean Squared ... " << std::endl;
    MultiLayerNeuralNetwork mlnnms(0.009f, MEAN_SQUARED);
    mlnnms.loadData();
    mlnnms.runEpochs(200);

    std::cout << "Starting Multi Layer Neural Network with Cross Entropy ... " << std::endl;
    MultiLayerNeuralNetwork mlnnce(0.0009f, CROSS_ENTROPY);
    mlnnce.loadData();
    mlnnce.runEpochs(200);
}

int main()
{
    CuBase::initializeHandle();
    run();
    CuBase::closeHandle();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    gpuErrchk(cudaDeviceReset());

    return 0;
}
