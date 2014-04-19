#pragma once
class CuBase
{
protected:
    static cublasHandle_t cuHandle;
public:
    static void initializeHandle();
    static void closeHandle();
};

template <class T>
class CuMatrix : CuBase 
{
    int d0;
    int d1;
    T *gpuData;


public:
    CuMatrix<T>(int d0, int d1);
    CuMatrix<T>(CuMatrix<T> &m);
    ~CuMatrix<T>(void);

    // Loads data from *data. Assumes that data is in a column-major format
    // and the shape of the data is exactly that of the matrix
    void loadDataFrom(T *data);

    // Loads the matrix data from the GPU
    T* returnData();

    // Transfer the GPU from a different CuMatrix
    void transferData(T *gpuData);

    // Performs the operation C = A + B
    static void add(CuMatrix<T> &a, CuMatrix<T> &b, CuMatrix<T> &c);

    // Performs the operation C = A + vec * [1,1,...,1]
    static void addVector(CuMatrix<T> &a, CuMatrix<T> &vec, CuMatrix<T> &c);

    // Performs the operation C = A * B
    static void multiply(CuMatrix<float> &a, bool trA, CuMatrix<float> &b, bool trB, CuMatrix<float> &c);

    // Performs the operation C = A x B where x is the Hadamard product
    static void hadm(CuMatrix<T> &a, CuMatrix<T> &b, CuMatrix<T> &c);
    
    // Apply the sigmoid function element-wise on all elements of the matrix
    void applySigmoid();

    // Sum all the values in the matrix and return the result
    T reduce();
};
