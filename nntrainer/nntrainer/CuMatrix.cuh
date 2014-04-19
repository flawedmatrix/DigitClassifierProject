#pragma once

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
    int d0;
    int d1;
    T *gpuData;

    CuMatrixBase();
    CuMatrixBase(int d0, int d1);
    CuMatrixBase(CuMatrixBase<T> &m);
    ~CuMatrixBase(void);

public:
    int getRows();
    int getCols();

    // Loads data from *data. Assumes that data is in a column-major format
    // and the shape of the data is exactly that of the matrix
    void loadDataFrom(T *data);

    // Loads the matrix data from the GPU
    T* returnData();

    // Transfer the gpuData from another source
    void transferData(T *gpuData);

    // Performs the operation C = A + B
    static void add(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c);

    // Performs the operation C = A - B
    static void sub(CuMatrixBase<T> &a, CuMatrixBase<T> &b, CuMatrixBase<T> &c);

    // Performs the operation C = A + vec * [1,1,...,1]
    static void addVector(CuMatrixBase<T> &a, CuMatrixBase<T> &vec, CuMatrixBase<T> &c);

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
    CuMatrix(int r, int c):CuMatrixBase<T>(r, c) {}
    CuMatrix(CuMatrix<T> &m):CuMatrixBase<T>(m) {}
};

template <>
class CuMatrix<int> : public CuMatrixBase<int> 
{
public:
    CuMatrix():CuMatrixBase<int>() {}
    CuMatrix(int r, int c):CuMatrixBase<int>(r, c) {}
    CuMatrix(CuMatrix<int> &m):CuMatrixBase<int>(m) {}

    // Performs the operation C = A != B for every element
    static void notEquals(CuMatrix<int> &a, CuMatrix<int> &b, CuMatrix<int> &c);
    // Populates a new float matrix from an int matrix
    void toFloat(CuMatrix<float> &target);
};

template <>
class CuMatrix<float> : public CuMatrixBase<float> 
{
public:
    CuMatrix():CuMatrixBase<float>() {}
    CuMatrix(int r, int c):CuMatrixBase<float>(r, c) {}
    CuMatrix(CuMatrix<float> &m):CuMatrixBase<float>(m) {}

    // Performs the operation C = A * B
    static void multiply(CuMatrix<float> &a, bool trA, CuMatrix<float> &b, bool trB, CuMatrix<float> &c);

    // Returns the index for which the value is the largest for each column of the matrix
    void argmax(CuMatrix<int> &out);

    // Apply the sigmoid function element-wise on all elements of the matrix
    void applySigmoid();

    // Performs the operation A = factor * A
    void scale(float factor);

    // Normalize all the values in the matrix to be between 0 and 1
    // Assumes all values are greater than 0
    void normalize(float max);

    // Assigns random values between 0 and 1 to all values of this matrix
    void initRandom();
};