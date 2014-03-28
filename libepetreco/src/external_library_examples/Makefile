EXAMPLES_SRC = .
CPPC = g++
override CPPCFLAGS += -Wall
INC = -I$(HDF5_ROOT)/include
LIB = -L$(HDF5_ROOT)/lib -lhdf5 -lhdf5_cpp

CUC = nvcc
CUCFLAGS =
CUCINC = -I$(CUDA_ROOT)/include
CUCLIB = -L$(CUDA_ROOT)/lib -lcublas

EXAMPLES := \
      example1_H5read.out \
      example2_H5read.out \
      example_H5write.out \
      example_H5CompoundType.out \
      example_cublas_MatrixVectorMult.out \
      example_cublas_MatrixVectorMult2.out \
      example_cublasxt_MatrixVectorMult.out \
      example_thrustZipIterator.out

default : $(EXAMPLES)

example%.out : \
      $(EXAMPLES_SRC)/example%.cpp
	$(CPPC) $(CPPCFLAGS) $(INC) $(LIB) $< -o $@

example%.out : \
      $(EXAMPLES_SRC)/example%.cu
	$(CUC) $(CUCFLAGS) $(CUCINC) $(CUCLIB) $< -o $@


clean :
	rm \
    $(EXAMPLES) \
    ./*.h5
