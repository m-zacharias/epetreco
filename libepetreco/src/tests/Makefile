override FLAGS +=

CUC = nvcc
CUCFLAGS = -arch=sm_20 $(FLAGS) -Xcompiler -Wall
#CUCFLAGS = -arch=sm_20 -Xcompiler -Wall -DCHORDSCALC_DEBUG

CPPC = g++
CPPCFLAGS = -Wall $(FLAGS)

INC = -I../\
      -I../../\
      -I$(HDF5_INC)\
      -I../Ply/

LIB = -L$(HDF5_LIB)\
      -lhdf5 -lhdf5_cpp\
      -lcublas\
      -L../../\
      -lepetreco

EXECUTABLES := \
      test_CudaMatrix.out\
      test_CudaTransform.out\
      test_CudaTransform2.out\
      test_CudaTransform3.out\
      test_CudaTransform4.out\
      test_CudaTransform5.out


default : \
      $(EXECUTABLES)

clean :
	rm $(EXECUTABLES) \
     ./*.ply \
     ./*.h5

run_tests : \
      $(EXECUTABLES)
	./test_CudaMatrix.out ; \
  ./test_CudaTransform.out ; \
  ./test_CudaTransform2.out ; \
  ./test_CudaTransform3.out ; \
  ./test_CudaTransform4.out ; \
  ./test_CudaTransform5.out

%.out : \
      ./%.cu
	CPLUS_INCLUDE_PATH= ; \
  $(CUC) $(CUCFLAGS) $(INC) $(LIB) $^ -o $@
