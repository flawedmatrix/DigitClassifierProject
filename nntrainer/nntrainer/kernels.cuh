#pragma once
#include "helpers.cuh"

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
__global__ void matrixSelectData(const T *A, const unsigned int *S, T *C, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements) {
        unsigned int dest = S[i1] * grid_width + i0;
        C[dest] = A[i];
    }
}

template <class T>
__global__ void matrixAdd(const T *A, const T *B, T *C, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

template <class T>
__global__ void matrixSub(const T *A, const T *B, T *C, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements) {
        C[i] = A[i] - B[i];
    }
}

template <class T>
__global__ void matrixAdd2(const T *A, const T *B, T *C, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements) {
        C[i] = A[i] + B[i0];
    }
}

__global__ void matrixNotEquals(const int *A, const int *B, int *C, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements) {
        C[i] = (int)(A[i] != B[i]);
    }
}

__global__ void matrixScale(float *A, float factor, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements) {
        A[i] = A[i] * factor;
    }
}

template <class T>
__global__ void matrixHadm(const T *A, const T *B, T *C, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements) {
        C[i] = A[i] * B[i];
    }
}

__global__ void matrixApplySigmoid(float *A, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements) {
        float z = A[i];
        float denom = 1 + exp(-z);
        A[i] = 1/denom;
    }
}
__global__ void matrixNormalize(float *A, float max, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements) {
        A[i] = A[i]/max;
    }
}

__global__ void convertToFloat(int *A, float *B, const size_t numElements) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    // map the two 2D indices to a single linear, 1D index
    unsigned int grid_width = gridDim.x * blockDim.x;
    unsigned int i = i1 * grid_width + i0;
    
    if (i < numElements) {
        B[i] = (float)A[i];
    }
}

template <class T>
__global__ void applyArgmax(T *A, int *B, const size_t rows, const size_t cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < cols) {
        int max_row = 0;
        float max_val = 0xff800000; // Minus infinity
        for (unsigned int j = 0; j < rows; j++) {
            int idx = j + i*rows;
            if ((float)A[idx] > max_val) {
                max_row = j;
                max_val = A[idx];
            }
        }
        B[i] = max_row;
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
