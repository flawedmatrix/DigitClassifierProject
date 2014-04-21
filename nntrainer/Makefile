CUDA_PATH	= /usr/local/cuda-4.2
CUDA_INC_PATH	= $(CUDA_PATH)/include
CUDA_BIN_PATH	= $(CUDA_PATH)/bin
CUDA_LIB_PATH	= $(CUDA_PATH)/lib64

NVCC		= $(CUDA_BIN_PATH)/nvcc

SOURCE_PATH	= nntrainer
FLAGS = -I$(CUDA_INC_PATH) -L$(CUDA_LIB_PATH) -lcurand -lcublas -lcudart

OBJECTS = CuMatrix.o NeuralNetwork.o SingleLayerNeuralNetwork.o main.o
TARGET = nntrain

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(FLAGS) $(OBJECTS) -o $(TARGET)
	
%.o : $(SOURCE_PATH)/%.cpp
	$(NVCC) $(FLAGS) --compiler-options "-std=c++0x" -c $<

%.o : $(SOURCE_PATH)/%.cu
	$(NVCC) $(FLAGS) -c $<

clean:
	rm *.o;rm $(TARGET)
