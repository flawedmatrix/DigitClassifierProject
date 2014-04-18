#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include "mat.h"

#include "helpers.cuh"

void computeForwardPass() {

}


int main()
{

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    gpuErrchk(cudaDeviceReset());

    return 0;
}