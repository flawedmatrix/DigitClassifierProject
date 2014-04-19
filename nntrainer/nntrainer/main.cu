#include "helpers.cuh"
#include <iostream>
#include "CuMatrix.cuh"

void run() {
    int num_elems = 60000;
    int *testData = new int[num_elems];
    for (int i = 0; i < num_elems; i++) {
        testData[i] = i;
    }
    int actual = 0;
    for (int i = 0; i < num_elems; i++) {
        actual += testData[i];
    }
    std::cout << "Expected " << actual << std::endl;
    CuMatrix<int> c = CuMatrix<int>(1, num_elems/2);
    CuMatrix<int> d = CuMatrix<int>(1, num_elems/2);
    CuMatrix<int> e = CuMatrix<int>(1, num_elems/2);
    c.loadDataFrom(testData);
    d.loadDataFrom(testData + num_elems/2);
    CuMatrix<int>::add(c, d, e);

    int result = e.reduce();

    std::cout << "Summed result " << result << std::endl;

    delete testData;
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