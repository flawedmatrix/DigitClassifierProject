#include "helpers.cuh"
#include "CuMatrix.cuh"
#include "kernels.cuh"
#include "curand.h"
#include "time.h"

cublasHandle_t CuBase::cuHandle = nullptr;

// Explicit declarations of the templated class
template class CuMatrixBase<int>;
template class CuMatrixBase<float>;

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
    d0(0), d1(0), gpuData(nullptr)
{
}
template <class T>
CuMatrixBase<T>::CuMatrixBase(int rows, int cols):
    d0(rows), d1(cols), gpuData(nullptr)
{
}

template <class T>
CuMatrixBase<T>::CuMatrixBase(CuMatrixBase<T> &m) {
    d0 = m.d0;
    d1 = m.d1;
    if (gpuData != nullptr) {
        gpuErrchk(cudaMalloc((void**)&gpuData, d0 * d1 * sizeof(T)));
        gpuErrchk(cudaMemcpy(gpuData, m.gpuData, d0 * d1 * sizeof(T), cudaMemcpyDeviceToDevice));
    }
}

template <class T>
CuMatrixBase<T>::~CuMatrixBase(void) {
    gpuErrchk(cudaFree(gpuData));
}
template <class T>
int CuMatrixBase<T>::getRows() {
    return d0;
}

template <class T>
int CuMatrixBase<T>::getCols() {
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
void CuMatrixBase<T>::add(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot add two dissimilar matrices";
    }
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));

    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixAdd<T><<<dimGrid, dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0 * a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

template <class T>
void CuMatrixBase<T>::sub(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot sub two dissimilar matrices";
    }
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));

    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixSub<T><<<dimGrid, dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0 * a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

template <class T>
void CuMatrixBase<T>::addVector(CuMatrixBase<T> &a, CuMatrixBase<T> &vec, CuMatrixBase<T> &c) {
    if (a.d0 != vec.d0) {
        throw "Cannot add matrices with different number of rows";
    }
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));
    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixAdd2<T><<<dimGrid, dimBlock>>>(a.gpuData, vec.gpuData, cData, a.d0 * a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

template <class T>
void CuMatrixBase<T>::hadm(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot hadm two dissimilar matrices";
    }
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));

    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixHadm<T><<<dimGrid, dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0 * a.d1);
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
    reduction<T><<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(gpuData, partial_sums, d0 * d1);
    // Launch a single block to compute sum of partial sums
    reduction<T><<<1, blocksPerGrid, blocksPerGrid * sizeof(T)>>>(partial_sums, partial_sums + blocksPerGrid, blocksPerGrid);

    T result = 0;
    gpuErrchk(cudaMemcpy(&result, partial_sums + blocksPerGrid, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(partial_sums));

    return result;
}


void CuMatrix<int>::notEquals(CuMatrix<int> &a, CuMatrix<int> &b, CuMatrix<int> &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot xor two dissimilar matrices";
    }
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));

    int *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(int)));
    matrixNotEquals<<<dimGrid, dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0 * a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

void CuMatrix<int>::toFloat(CuMatrix<float> &target) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));

    float *tData;
    gpuErrchk(cudaMalloc((void**)&tData, d0 * d1 * sizeof(float)));
    convertToFloat<<<dimGrid, dimBlock>>>(gpuData, tData, d0 * d1);
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
    // Do the actual multiplication
    cublasSgemm(cuHandle, opA, opB, a.d0, b.d1, a.d1, alpha, a.gpuData, a.d0, b.gpuData, b.d0, beta, c.gpuData, c.d0);
}

void CuMatrix<float>::applySigmoid() {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));
    matrixApplySigmoid<<<dimGrid, dimBlock>>>(gpuData, d0 * d1);
}

void CuMatrix<float>::argmax(CuMatrix<int> &out) {
    // Spawn one thread per column of the matrix
    unsigned int threadsPerBlock = 512;
    unsigned int blocksPerGrid = (d0 * d1 + threadsPerBlock - 1) / threadsPerBlock;

    int *tData;
    gpuErrchk(cudaMalloc((void**)&tData, d1 * sizeof(int)));
    applyArgmax<float><<<blocksPerGrid, threadsPerBlock>>>(gpuData, tData, d0, d1);
    gpuErrchk(cudaGetLastError());
    out.transferData(tData);
}

void CuMatrix<float>::scale(float factor) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));
    matrixScale<<<dimGrid, dimBlock>>>(gpuData, factor, d0 * d1);
}

void CuMatrix<float>::normalize(float max) {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));
    matrixNormalize<<<dimGrid, dimBlock>>>(gpuData, max, d0 * d1);
}

void CuMatrix<float>::initRandom() {
    float *tData;
    gpuErrchk(cudaMalloc((void**)&tData, d0 * d1 * sizeof(int)));
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    CURAND_CALL(curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT));

    // Set the seed for the random number generator using the system clock
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock()));
    // Fill the array with random numbers on the device
    CURAND_CALL(curandGenerateUniform(prng, tData, d0 * d1));
    transferData(tData);

}