#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>

#include "helpers.cuh"
#include "CuMatrix.cuh"


__global__ void matrixAdd(const float *A, const float *B, float *C, int numElements)
{
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    int grid_width = gridDim.x * blockDim.x;
    int i = i1 * grid_width + i0;
    
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}
__global__ void matrixAdd2(const float *A, const float *B, float *C, int numElements)
{
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    int grid_width = gridDim.x * blockDim.x;
    int i = i1 * grid_width + i0;
    
    if (i < numElements)
    {
        C[i] = A[i] + B[i0];
    }
}

__global__ void matrixHadm(const float *A, const float *B, float *C, int numElements)
{
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    int grid_width = gridDim.x * blockDim.x;
    int i = i1 * grid_width + i0;
    
    if (i < numElements)
    {
        C[i] = A[i] * B[i];
    }
}

__global__ void applySigmoid(float *A, int numElements)
{
    int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    int grid_width = gridDim.x * blockDim.x;
    int i = i1 * grid_width + i0;
    
    if (i < numElements)
    {
        float z = A[i];
        float denom = 1 + exp(-z);
        A[i] = 1/denom;
    }
}

cublasHandle_t CuMatrix::cuHandle = nullptr;

CuMatrix::CuMatrix(int rows, int cols):
    d0(rows), d1(cols), gpuData(0)
{
}

CuMatrix::CuMatrix(CuMatrix &m) {
    d0 = m.d0;
    d1 = m.d1;
    gpuErrchk(cudaMalloc((void**)&gpuData, d0 * d1 * sizeof(float)));
    gpuErrchk(cudaMemcpy(gpuData, m.gpuData, d0 * d1 * sizeof(int), cudaMemcpyDeviceToDevice));
}

CuMatrix::~CuMatrix(void) {
    gpuErrchk(cudaFree(gpuData));
}

void CuMatrix::initializeHandle() {
    // Create a handle for CUBLAS
    cublasCreate(&cuHandle);
}

void CuMatrix::closeHandle() {
    // Destroy the handle
    cublasDestroy(cuHandle);
}

void CuMatrix::loadDataFrom(float *data) {
    // Malloc some GPU memory
    gpuErrchk(cudaMalloc((void**)&gpuData, d0 * d1 * sizeof(float)));
    // Copy the data from the data buffer to the device
    gpuErrchk(cudaMemcpy(gpuData, data, d0 * d1 * sizeof(int), cudaMemcpyHostToDevice));
}

void CuMatrix::transferData(float *gpuData) {
    gpuErrchk(cudaFree(gpuData));
    this->gpuData = gpuData;
}

// Performs the operation C = A + B
void CuMatrix::add(CuMatrix &a, CuMatrix &b, CuMatrix &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot add two dissimilar matrices";
    }
    dim3 dimBlock(256, 256);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));

    float *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(float)));
    matrixAdd<<<dimGrid, dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0 * a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

// Performs the operation C = A + vec * [1,1,...,1]
void CuMatrix::addVector(CuMatrix &a, CuMatrix &vec, CuMatrix &c) {
    if (a.d0 != vec.d0) {
        throw "Cannot add matrices with different number of rows";
    }
    dim3 dimBlock(256, 256);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));
    float *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(float)));
    matrixAdd2<<<dimGrid, dimBlock>>>(a.gpuData, vec.gpuData, cData, a.d0 * a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

// Performs the operation C = A * B
void CuMatrix::multiply(CuMatrix &a, bool trA, CuMatrix &b, bool trB, CuMatrix &c) {
    if ((a.d0 != c.d0) || (b.d1 != c.d1) || (a.d1 != b.d0)) {
        throw "Matrix dimensions not correct";
    }

    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cublasOperation_t opA = trA? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = trB? CUBLAS_OP_T : CUBLAS_OP_N;
    // Do the actual multiplication
    cublasSgemm(cuHandle, opA, opB, a.d0, b.d1, a.d1, alpha, a.gpuData, a.d0, b.gpuData, b.d0, beta, c.gpuData, c.d0);
}

// Performs the operation C = A x B where x is the Hadamard product
void CuMatrix::hadm(CuMatrix &a, CuMatrix &b, CuMatrix &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot hadm two dissimilar matrices";
    }
    dim3 dimBlock(256, 256);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));

    float *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(float)));
    matrixHadm<<<dimGrid, dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0 * a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}
    
void CuMatrix::applySigmoid() {
    dim3 dimBlock(256, 256);
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));

    float *cData;
    gpuErrchk(cudaMalloc((void**)&cData, d0 * d1 * sizeof(float)));
    matrixHadm<<<dimGrid, dimBlock>>>(gpuData, d0 * d1);
    gpuErrchk(cudaGetLastError());
}

