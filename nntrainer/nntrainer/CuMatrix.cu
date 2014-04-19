#include "helpers.cuh"
#include "CuMatrix.cuh"

// non-specialized class template
template <class T>
class SharedMem
{
public:
    // Ensure that we won't compile any un-specialized types
    T* getPointer() { exit(1); };
};

// specialization for int
template <>
class SharedMem <int>
{
public:
    __device__ int* getPointer() { extern __shared__ int s_int[]; return s_int; }
};

// specialization for float
template <>
class SharedMem <float>
{
public:
    __device__ float* getPointer() { extern __shared__ float s_float[]; return s_float; }
};

template <class T>
__global__ void matrixAdd(const T *A, const T *B, T *C, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

template <class T>
__global__ void matrixAdd2(const T *A, const T *B, T *C, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements)
    {
        C[i] = A[i] + B[i0];
    }
}

template <class T>
__global__ void matrixHadm(const T *A, const T *B, T *C, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements)
    {
        C[i] = A[i] * B[i];
    }
}

__global__ void matrixApplySigmoid(float *A, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements)
    {
        float z = A[i];
        float denom = 1 + exp(-z);
        A[i] = 1/denom;
    }
}

template <class T>
__global__ void reduction(const T *A, T *B, const size_t numElements) {
    SharedMem<T> shared;
    T* sdata = shared.getPointer();

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    T x = 0;
    if (i < numElements) {
        x = A[i];
    }
    // each thread loads one element from global to shared mem
    sdata[tid] = x;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        // wait until all threads in the block have updated their partial sums
        __syncthreads();
    }
    // thread 0 writes the per-block result
    if (tid == 0) B[blockIdx.x] = sdata[0];
}

cublasHandle_t CuBase::cuHandle = nullptr;

template class CuMatrix<int>;
template class CuMatrix<float>;

void CuBase::initializeHandle() {
    // Create a handle for CUBLAS
    cublasCreate(&cuHandle);
}

void CuBase::closeHandle() {
    // Destroy the handle
    cublasDestroy(cuHandle);
}

template <class T>
CuMatrix<T>::CuMatrix(int rows, int cols):
    d0(rows), d1(cols), gpuData(0)
{
}

template <class T>
CuMatrix<T>::CuMatrix(CuMatrix<T> &m) {
    d0 = m.d0;
    d1 = m.d1;
    gpuErrchk(cudaMalloc((void**)&gpuData, d0 * d1 * sizeof(T)));
    gpuErrchk(cudaMemcpy(gpuData, m.gpuData, d0 * d1 * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <class T>
CuMatrix<T>::~CuMatrix(void) {
    gpuErrchk(cudaFree(gpuData));
}


template <class T>
void CuMatrix<T>::loadDataFrom(T *data) {
    // Malloc some GPU memory
    gpuErrchk(cudaMalloc((void**)&gpuData, d0 * d1 * sizeof(T)));
    // Copy the data from the data buffer to the device
    gpuErrchk(cudaMemcpy(gpuData, data, d0 * d1 * sizeof(T), cudaMemcpyHostToDevice));
}

template <class T>
T* CuMatrix<T>::returnData() {
    T* data = new T[d0*d1];
    // Copy the data from the device to the data buffer
    gpuErrchk(cudaMemcpy(data, gpuData, d0 * d1 * sizeof(T), cudaMemcpyDeviceToHost));
    return data;
}

template <class T>
void CuMatrix<T>::transferData(T *newData) {
    gpuErrchk(cudaFree(gpuData));
    gpuData = newData;
}

// Performs the operation C = A + B
template <class T>
void CuMatrix<T>::add(CuMatrix<T> &a, CuMatrix<T> &b, CuMatrix<T> &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot add two dissimilar matrices";
    }
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));

    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixAdd<<<dimGrid, dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0 * a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

// Performs the operation C = A + vec * [1,1,...,1]
template <class T>
void CuMatrix<T>::addVector(CuMatrix<T> &a, CuMatrix<T> &vec, CuMatrix<T> &c) {
    if (a.d0 != vec.d0) {
        throw "Cannot add matrices with different number of rows";
    }
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));
    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixAdd2<<<dimGrid, dimBlock>>>(a.gpuData, vec.gpuData, cData, a.d0 * a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}

// Performs the operation C = A * B
template <>
void CuMatrix<float>::multiply(CuMatrix<float> &a, bool trA, CuMatrix<float> &b, bool trB, CuMatrix<float> &c) {
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
template <class T>
void CuMatrix<T>::hadm(CuMatrix<T> &a, CuMatrix<T> &b, CuMatrix<T> &c) {
    if ((a.d0 != b.d0) || (a.d1 != b.d1)) {
        throw "Cannot hadm two dissimilar matrices";
    }
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)a.d0/dimBlock.x),(int)ceil((float)a.d1/dimBlock.y));

    T *cData;
    gpuErrchk(cudaMalloc((void**)&cData, a.d0 * a.d1 * sizeof(T)));
    matrixHadm<<<dimGrid, dimBlock>>>(a.gpuData, b.gpuData, cData, a.d0 * a.d1);
    gpuErrchk(cudaGetLastError());
    c.transferData(cData);
}
    
template <>
void CuMatrix<float>::applySigmoid() {
    dim3 dimBlock(32, 32);
    dim3 dimGrid((int)ceil((float)d0/dimBlock.x),(int)ceil((float)d1/dimBlock.y));
    matrixApplySigmoid<<<dimGrid, dimBlock>>>(gpuData, d0 * d1);
}

template <class T>
T CuMatrix<T>::reduce() {
    unsigned int threadsPerBlock = 512;
    unsigned int blocksPerGrid = (d0 * d1 + threadsPerBlock - 1) / threadsPerBlock;
    blocksPerGrid = nextpo2(blocksPerGrid);

    T *partial_sums = 0;
    gpuErrchk(cudaMalloc((void**)&partial_sums, (blocksPerGrid + 1) * sizeof(T)));
    // Compute partial sums for all blocks
    reduction<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(T)>>>(gpuData, partial_sums, d0 * d1);
    // Launch a single block to compute sum of partial sums
    reduction<<<1, blocksPerGrid, blocksPerGrid * sizeof(T)>>>(partial_sums, partial_sums + blocksPerGrid, blocksPerGrid);

    T result = 0;
    gpuErrchk(cudaMemcpy(&result, partial_sums + blocksPerGrid, sizeof(T), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(partial_sums));

    return result;
}