#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "cublas_v2.h"

class CuBase
{
protected:
    static cublasHandle_t cuHandle;
public:
    static void initializeHandle();
    static void closeHandle();
};

template <class T>
class CuMatrixBase : protected CuBase
{
protected:
    size_t d0;
    size_t d1;
    T *gpuData;
    unsigned int* selection;
    dim3 dimBlock;

    CuMatrixBase();
    CuMatrixBase(size_t r, size_t c);
    CuMatrixBase(const CuMatrixBase<T> &m);
    ~CuMatrixBase(void);

public:
    size_t getRows();
    size_t getCols();

    // Loads data from *data. Assumes that data is in a column-major format
    // and the shape of the data is exactly that of the matrix
    void loadDataFrom(T *data);

    // Loads the selection indices for use when selectData is called
    void loadSelection(unsigned int *h_selection);

    // Selects n columns indexed by selection from this matrix
    void selectData(CuMatrixBase<T> &out, unsigned int blockStart, size_t n);

    // Loads the matrix data from the GPU
    T* returnData();

    // Transfer the gpuData from another source
    void transferData(T *gpuData);

    // Fill the matrix with num
    void fill(T num);

    // Performs the operation C = A + B
    static void add(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c);

    // Performs the operation C = A - B
    static void sub(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c);

    // Performs the operation C = A + vec * [1,1,...,1]
    static void addVector(CuMatrixBase<T> &a, CuMatrixBase<T> &vec, CuMatrixBase<T> &c);

    // Performs the operation C = A - vec * [1,1,...,1]
    static void subVector(CuMatrixBase<T> &a, CuMatrixBase<T> &vec, CuMatrixBase<T> &c);

    // Performs the operation C = A x B where x is the Hadamard product
    static void hadm(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c);

    // Sum all the values in the matrix and return the result
    T reduce();
};

template <class T>
class CuMatrix : CuMatrixBase<T> 
{
public:
    CuMatrix():CuMatrixBase<T>() {}
    CuMatrix(size_t r, size_t c):CuMatrixBase<T>(r, c) {}
    CuMatrix(const CuMatrix<T> &m):CuMatrixBase<T>(m) {}
};

template <>
class CuMatrix<char> : public CuMatrixBase<char> 
{
public:
    CuMatrix():CuMatrixBase<char>() {}
    CuMatrix(size_t r, size_t c):CuMatrixBase<char>(r, c) {}
    CuMatrix(const CuMatrix<char> &m):CuMatrixBase<char>(m) {}

    // Performs the operation C = A != B for every element
    static void notEquals(CuMatrix<char> &a, CuMatrix<char> &b, CuMatrix<char> &c);
    // Populates a new float matrix from an int matrix
    void toFloat(CuMatrix<float> &target);

    // For each element of the matrix, create an d tall column where it is 1 where
    // row# = value and 0 otherwise
    void encode(CuMatrix<float> &out, size_t d);

    // Specialized int output reduction for char
    int reduce();
};

template <>
class CuMatrix<float> : public CuMatrixBase<float> 
{
public:
    CuMatrix():CuMatrixBase<float>() {}
    CuMatrix(size_t r, size_t c):CuMatrixBase<float>(r, c) {}
    CuMatrix(const CuMatrix<float> &m):CuMatrixBase<float>(m) {}

    // Performs the operation C = A * B
    static void multiply(CuMatrix<float> &a, bool trA, CuMatrix<float> &b, bool trB, CuMatrix<float> &c);

    // Performs the operation C = A / vec * [1,1,...,1]
    // Where the division is an element wise divide
    static void divVector(CuMatrix<float> &a, CuMatrix<float> &vec, CuMatrix<float> &c);

    // Returns the index for which the value is the largest for each column of the matrix
    void argmax(CuMatrix<char> &out);

    // Apply the sigmoid function element-wise on all elements of the matrix
    void applySigmoid();

    // Apply the tanh function element-wise on all elements of the matrix
    void applyTanh();
    
    // Apply the sqrt function element-wise on all elements of the matrix
    void applySqrt();

    // Performs the operation A = factor * A
    void scale(float factor);

    // Normalize all the values in the matrix to be between 0 and 1
    // Assumes all values are greater than 0
    void normalize(float max);

    // Standardizes the values used in the matrix
    void standardize();

    // Assigns random values between 0 and 1 to all values of this matrix
    void initRandom();
};