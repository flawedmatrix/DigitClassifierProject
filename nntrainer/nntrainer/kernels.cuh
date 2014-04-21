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

// specialization for char
template <>
class SharedMem <char>
{
public:
    __device__ char* getPointer() { extern __shared__ char s_char[]; return s_char; }
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
__global__ void matrixSelectData(const T *A, const unsigned int *S, T *C, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        unsigned int src = S[i1] * d + i0;
        C[i] = A[src];
    }
}

template <class T>
__global__ void matrixFill(T *A, T num, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        A[i] = num;
    }
}

template <class T>
__global__ void matrixAdd(const T *A, const T *B, T *C, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        C[i] = A[i] + B[i];
    }
}

template <class T>
__global__ void matrixSub(const T *A, const T *B, T *C, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

     if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        C[i] = A[i] - B[i];
    }
}

template <class T>
__global__ void matrixAdd2(const T *A, const T *B, T *C, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        C[i] = A[i] + B[i0];
    }
}

__global__ void matrixNotEquals(const char *A, const char *B, char *C, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        C[i] = (char)(A[i] != B[i]);
    }
}

__global__ void matrixScale(float *A, float factor, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        A[i] = A[i] * factor;
    }
}

template <class T>
__global__ void matrixHadm(const T *A, const T *B, T *C, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        C[i] = A[i] * B[i];
    }
}

__global__ void matrixApplySigmoid(float *A, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;
        
    if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        float z = A[i];
        float denom = 1 + exp(-z);
        A[i] = 1.0f/denom;
    }
}
__global__ void matrixNormalize(float *A, float max, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        float z = A[i]/max;
        A[i] = z;
    }
}

__global__ void matrixEncode(char *A, float *B, const size_t d, const size_t d0, const size_t d1) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < d0 && i1 < d1) {
        unsigned int i = i1 * d0 + i0;
        unsigned int bi = i * d + A[i];
        B[bi] = 1;
    }
}

__global__ void convertToFloat(char *A, float *B, const size_t d, const size_t n) {
    unsigned int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int i1 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i0 < d && i1 < n) {
        unsigned int i = i1 * d + i0;
        B[i] = (float)(unsigned char)A[i];
    }
}

template <class T>
__global__ void applyArgmax(T *A, char *B, const size_t rows, const size_t cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < cols) {
        unsigned int max_row = 255;
        float max_val = -999999999.0f; // Minus infinity
        for (unsigned int j = 0; j < rows; j++) {
            unsigned int idx = j + i*rows;
            if ((float)A[idx] > max_val) {
                max_row = j;
                max_val = A[idx];
            }
        }
        B[i] = (char)max_row;
    }
}

template <class T, class R>
__global__ void reduction(const T *A, R *B, const size_t numElements) {
    SharedMem<R> shared;
    R* sdata = shared.getPointer();

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + tid;
    T x = 0;
    if (i < numElements) {
        x = A[i];
    }
    // each thread loads one element from global to shared mem
    sdata[tid] = (R)x;
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
