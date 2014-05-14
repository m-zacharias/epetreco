CUC = nvcc
CUCFLAGS = -arch=sm_35 -Xcompiler -Wall -DSPECIAL

CPPC = g++
CPPCFLAGS = -Wall

INC = -I../ -I../../ -I$(HDF5_INC) -I../Ply/
LIB = -L$(HDF5_LIB) -lhdf5 -lhdf5_cpp -lcublas -L../../ -lepetreco

real_algo.out : \
      real_algo.cu
	CPLUS_INCLUDE_PATH=;\
  $(CUC) $(CUCFLAGS) $(INC) $(LIB) $^ -o $@

algo.out : \
      algo.cu
	CPLUS_INCLUDE_PATH=;\
  $(CUC) $(CUCFLAGS) $(INC) $(LIB) $^ -o $@



clean :
	rm ./algo.out ./real_algo.out
