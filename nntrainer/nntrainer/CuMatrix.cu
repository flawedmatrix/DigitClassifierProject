#include "helpers.cuh"
#include "CuMatrix.cuh"
#include "kernels.cuh"
#include "curand.h"
#include <time.h>

cublasHandle_t CuBase::cuHandle = NULL;

void CuBase::initializeHandle() {
    // Create a handle for CUBLAS
    cublasCreate(&cuHandle);
}

void CuBase::closeHandle() {
    // Destroy the handle
    cublasDestroy(cuHandle);
}

template <class T>
CuMatrixBase<T>::CuMatrixBase():
    d0(0), d1(0), gpuData(NULL), selection(NULL)
{
    dimBlock = dim3(32, 32);
}
template <class T>
CuMatrixBase<T>::CuMatrixBase(size_t rows, size_t cols):
    d0(rows), d1(cols), gpuData(NULL), selection(NULL)
{
    if (rows * 2 < cols) {
        dimBlock = dim3(4, 256);
    } else if (rows > cols * 2) {
        dimBlock = dim3(256, 4);
    } else {
        dimBlock = dim3(32, 32);
    }
}

template <class T>
CuMatrixBase<T>::CuMatrixBase(const CuMatrixBase<T> &m) {
    d0 = m.d0;
    d1 = m.d1;
    if (gpuData != NULL) {
        gpuErrchk(cudaMalloc((void**)&gpuData, d0 * d1 * sizeof(T)));
        gpuErrchk(cudaMemcpy(gpuData, m.gpuData, d0 * d1 * sizeof(T), cudaMemcpyDeviceToDevice));
    }
    selection = NULL;
}

template <class T>
CuMatrixBase<T>::~CuMatrixBase(void) {
    gpuErrchk(cudaFree(gpuData));
    gpuErrchk(cudaFree(selection));
}
template <class T>
size_t CuMatrixBase<T>::getRows() {
    return d0;
}

template <class T>
size_t CuMatrixBase<T>::getCols() {
    return d1;
}

template <class T>
void CuMatrixBase<T>::loadDataFrom(T *data) {
    // Malloc some GPU memory
    gpuErrchk(cudaMalloc((void**)&gpuData, d0 * d1 * sizeof(T)));
    // Copy the data from the data buffer to the device
    gpuErrchk(cudaMemcpy(gpuData, data, d0 * d1 * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
void CuMatrixBase<T>::loadSelection(unsigned int *h_selection) { 
    gpuErrchk(cudaMalloc((void**)&selection, d1 * sizeof(unsigned int)));
    gpuErrchk(cudaMemcpy(selection, h_selection, d1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
}

template <class T>
void CuMatrixBase<T>::selectData(CuMatrixBase<T> &out, unsigned int blockStart, size_t n) {
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)n/dimBlock.y));

    T *tData;
    gpuErrchk(cudaMalloc((void**)&tData, d0 * n * sizeof(T)));
    matrixSelectData<T><<<dimGrid, dimBlock>>>(gpuData, selection + blockStart, tData, d0, n);
    gpuErrchk(cudaGetLastError());
    
    out.transferData(tData);
}

template <class T>
T* CuMatrixBase<T>::returnData() {
    T* data = new T[d0*d1];
    // Copy the data from the device to the data buffer
    gpuErrchk(cudaMemcpy(data, gpuData, d0 * d1 * sizeof(T), cudaMemcpyDeviceToHost));
    return data;
}

template <class T>
void CuMatrixBase<T>::transferData(T *newData) {
    gpuErrchk(cudaFree(gpuData));
    gpuData = newData;
}

template <class T>
void CuMatrixBase<T>::fill(T num) {
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));

    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, d0 * d1 * sizeof(T)));
    matrixFill<T><<<dimGrid, dimBlock>>>(cData, num, d0, d1);
    gpuErrchk(cudaGetLastError());
    transferData(cData);
}

template <class T>
void CuMatrixBase<T>::add(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot add two dissimilar matrices";
    }
    dim3 dimGrid((int)ceil((float)a.d0/a.dimBlock.x),(int)ceil((float)a.d1/a.dimBlock.y));

    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixAdd<T><<<dimGrid, a.dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0, a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

template <class T>
void CuMatrixBase<T>::sub(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot sub two dissimilar matrices";
    }
    dim3 dimGrid((int)ceil((float)a.d0/a.dimBlock.x),(int)ceil((float)a.d1/a.dimBlock.y));

    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixSub<T><<<dimGrid, a.dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0, a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

template <class T>
void CuMatrixBase<T>::addVector(CuMatrixBase<T> &a, CuMatrixBase<T> &vec, CuMatrixBase<T> &c) {
    if (a.d0 != vec.d0) {
        throw "Cannot add matrices with different number of rows";
    }
    dim3 dimGrid((int)ceil((float)a.d0/a.dimBlock.x),(int)ceil((float)a.d1/a.dimBlock.y));
    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixAdd2<T><<<dimGrid, a.dimBlock>>>(a.gpuData, vec.gpuData, cData, a.d0, a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

template <class T>
void CuMatrixBase<T>::hadm(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot hadm two dissimilar matrices";
    }
    dim3 dimGrid((int)ceil((float)a.d0/a.dimBlock.x),(int)ceil((float)a.d1/a.dimBlock.y));

    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixHadm<T><<<dimGrid, a.dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0, a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

template <class T>
T CuMatrixBase<T>::reduce() {
    unsigned int threadsPerBlock = 512;
    unsigned int blocksPerGrid = (d0 * d1 + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = nextpo2(blocksPerGrid);

    T *partial_sums = 0;
    gpuErrchk(cudaMalloc((void**)&partial_sums, (blocksPerGrid + 1) * sizeof(T)));
    // Compute partial sums for all blocks
    reduction<T, T><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(gpuData, partial_sums, d0 * d1);
    gpuErrchk(cudaGetLastError());
    // Launch a single block to compute sum of partial sums
    reduction<T, T><<<1, blocksPerGrid, blocksPerGrid * sizeof(T)>>>(partial_sums, partial_sums + blocksPerGrid, blocksPerGrid);
    gpuErrchk(cudaGetLastError());

    T result = 0;
    gpuErrchk(cudaMemcpy(&result, partial_sums + blocksPerGrid, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(partial_sums));

    return result;
}

int CuMatrix<char>::reduce() {
    unsigned int threadsPerBlock = 512;
    unsigned int blocksPerGrid = (d0 * d1 + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = nextpo2(blocksPerGrid);

    int *partial_sums = 0;
    gpuErrchk(cudaMalloc((void**)&partial_sums, (blocksPerGrid + 1) * sizeof(int)));
    // Compute partial sums for all blocks
    reduction<char, int><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(gpuData, partial_sums, d0 * d1);
    gpuErrchk(cudaGetLastError());
    // Launch a single block to compute sum of partial sums
    reduction<int, int><<<1, blocksPerGrid, blocksPerGrid * sizeof(int)>>>(partial_sums, partial_sums + blocksPerGrid, blocksPerGrid);
    gpuErrchk(cudaGetLastError());

    int result = 0;
    gpuErrchk(cudaMemcpy(&result, partial_sums + blocksPerGrid, sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(partial_sums));

    return result;
}

void CuMatrix<char>::encode(CuMatrix<float> &out, size_t d) {
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));

    float *tData;
    gpuErrchk(cudaMalloc((void**)&tData, d * d0 * d1 * sizeof(float)));
    gpuErrchk(cudaMemset(tData, 0, d * d0 * d1 * sizeof(float)));
    matrixEncode<<<dimGrid, dimBlock>>>(gpuData, tData, d, d0, d1);
    gpuErrchk(cudaGetLastError());
    out.transferData(tData);
}

void CuMatrix<char>::notEquals(CuMatrix<char> &a, CuMatrix<char> &b, CuMatrix<char> &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot xor two dissimilar matrices";
    }
    dim3 dimGrid((int)ceil((float)a.d0/a.dimBlock.x),(int)ceil((float)a.d1/a.dimBlock.y));

    char *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(char)));
    matrixNotEquals<<<dimGrid, a.dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0, a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

void CuMatrix<char>::toFloat(CuMatrix<float> &target) {
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));

    float *tData;
    gpuErrchk(cudaMalloc((void**)&tData, d0 * d1 * sizeof(float)));
    convertToFloat<<<dimGrid, dimBlock>>>(gpuData, tData, d0, d1);
    gpuErrchk(cudaGetLastError());
    target.transferData(tData);
}

void CuMatrix<float>::multiply(CuMatrix<float> &a, bool trA, CuMatrix<float> &b, bool trB, CuMatrix<float> &c) {
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cublasOperation_t opA = trA? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = trB? CUBLAS_OP_T : CUBLAS_OP_N;
    unsigned int m = c.d0;
    unsigned int n = c.d1;
    unsigned int k = trA? a.d0 : a.d1;
    
    float *cData;
    gpuErrchk(cudaMalloc((void**)&cData, m * n * sizeof(float)));
    // Do the actual multiplication
    CUBLAS_CALL(cublasSgemm_v2(cuHandle, opA, opB, m, n, k, alpha, a.gpuData, a.d0, b.gpuData, b.d0, beta, cData, c.d0));
    c.transferData(cData);
}

void CuMatrix<float>::applySigmoid() {
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));
    matrixApplySigmoid<<<dimGrid, dimBlock>>>(gpuData, d0, d1);
    gpuErrchk(cudaGetLastError());
}

void CuMatrix<float>::argmax(CuMatrix<char> &out) {
    // Spawn one thread per column of the matrix
    unsigned int threadsPerBlock = 512;
    unsigned int blocksPerGrid = (d0 * d1 + threadsPerBlock - 1) / threadsPerBlock;

    char *tData;
    gpuErrchk(cudaMalloc((void**)&tData, d1 * sizeof(int)));
    applyArgmax<float><<<blocksPerGrid, threadsPerBlock>>>(gpuData, tData, d0, d1);
    gpuErrchk(cudaGetLastError());
    out.transferData(tData);
}

void CuMatrix<float>::scale(float factor) {
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));
    matrixScale<<<dimGrid, dimBlock>>>(gpuData, factor, d0, d1);
    gpuErrchk(cudaGetLastError());
}

void CuMatrix<float>::normalize(float max) {
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));
    matrixNormalize<<<dimGrid, dimBlock>>>(gpuData, max, d0, d1);
    gpuErrchk(cudaGetLastError());
}

void CuMatrix<float>::initRandom() {
    float *tData;
    gpuErrchk(cudaMalloc((void**)&tData, d0 * d1 * sizeof(float)));
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    CURAND_CALL(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));

    // Set the seed for the random number generator using the system clock
    unsigned long long seed = (unsigned long long)clock();
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(prng, seed));
    // Fill the array with random numbers on the device
    CURAND_CALL(curandGenerateUniform(prng, tData, d0 * d1));
    transferData(tData);
    CURAND_CALL(curandDestroyGenerator(prng));
}

// Explicit declarations of the templated class
template class CuMatrixBase<char>;
template class CuMatrixBase<float>;