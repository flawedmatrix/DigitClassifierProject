#include "constants.h"
#include "helpers.cuh"
#include <iostream>
#include "CuMatrix.cuh"
#include "NeuralNetwork.h"
#include "SingleLayerNeuralNetwork.h"
#include "MultiLayerNeuralNetwork.h"

void run() {
    //std::cout << "Starting Single Layer Neural Network ... " << std::endl;
    //SingleLayerNeuralNetwork slnn(0.1f, MEAN_SQUARED);
    //slnn.loadData();
    //slnn.runEpochs(200);
    std::cout << "Starting Multi Layer Neural Network ... " << std::endl;
    MultiLayerNeuralNetwork mlnn(0.009f, MEAN_SQUARED);
    mlnn.loadData();
    mlnn.runEpochs(200);
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