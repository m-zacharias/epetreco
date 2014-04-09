SRC = ..
EXAMPLES_SRC = .
CPPC = g++
override CPPCFLAGS += -Wall
INC = -I$(HDF5_ROOT)/include -I$(SRC)
LIB = -L$(HDF5_ROOT)/lib -lhdf5 -lhdf5_cpp

CUC = nvcc
CUCFLAGS = -arch=sm_35
CUCINC = -I$(CUDA_ROOT)/include -I$(SRC)
CUCLIB = -L$(CUDA_ROOT)/lib -lcublas

EXAMPLES := \
      example1_H5read.out \
      example2_H5read.out \
      example_H5write.out \
      example_H5CompoundType.out \
      example_cublas_MatrixVectorMult.out \
      example_cublas_MatrixVectorMult2.out \
      example_cublasxt_MatrixVectorMult.out \
      example_thrust_zipIterator.out \
      example_curand.out \
      example_cuda_memory.out \
      example_cuda_double-buffering.out

default : $(EXAMPLES)

example%.out : \
      $(EXAMPLES_SRC)/example%.cpp
	$(CPPC) $(CPPCFLAGS) $(INC) $(LIB) $< -o $@

example%.out : \
      $(EXAMPLES_SRC)/example%.cu
	CPLUS_INCLUDE_PATH= ; \
  $(CUC) $(CUCFLAGS) $(CUCINC) $(CUCLIB) $< -o $@


clean :
	rm \
    $(EXAMPLES) \
    ./*.h5
