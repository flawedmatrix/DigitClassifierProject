
class CuMatrix
{
    int d0;
    int d1;
    float *gpuData;

    static cublasHandle_t cuHandle;
public:
    CuMatrix(int d0, int d1);
    CuMatrix(CuMatrix &m);
    ~CuMatrix(void);

    static void initializeHandle();
    static void closeHandle();

    // Loads data from *data. Assumes that data is in a column-major format
    // and the shape of the data is exactly that of the matrix
    void loadDataFrom(float *data);
    void transferData(float *gpuData);
    // Performs the operation C = A + B
    static void add(CuMatrix &a, CuMatrix &b, CuMatrix &c);
    // Performs the operation C = A + vec * [1,1,...,1]
    static void addVector(CuMatrix &a, CuMatrix &vec, CuMatrix &c);
    // Performs the operation C = A * B
    static void multiply(CuMatrix &a, bool trA, CuMatrix &b, bool trB, CuMatrix &c);
    // Performs the operation C = A x B where x is the Hadamard product
    static void hadm(CuMatrix &a, CuMatrix &b, CuMatrix &c);
    
    // Apply the sigmoid function element-wise on all elements of the matrix
    void applySigmoid();
};

