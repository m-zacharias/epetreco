CUC = nvcc
CUCFLAGS = -arch=sm_20 -Xcompiler -Wall
#CUCFLAGS = -arch=sm_20 -Xcompiler -Wall -DCHORDSCALC_DEBUG

CPPC = g++
CPPCFLAGS = -Wall

INC = -I../ -I../../ -I$(HDF5_INC) -I../Ply/
LIB = -L$(HDF5_LIB) -lhdf5 -lhdf5_cpp -lcublas -L../../ -lepetreco

EXECUTABLES := \
      real_algo.out \
      algo.out



default : \
      $(EXECUTABLES)

clean :
	rm $(EXECUTABLES) \
     ./*.ply \
     ./*.h5


%.out : \
      ./%.cu
	CPLUS_INCLUDE_PATH= ; \
  $(CUC) $(CUCFLAGS) $(INC) $(LIB) $^ -o $@

#real_algo.out : \
#      real_algo.cu
#	CPLUS_INCLUDE_PATH=;\
#  $(CUC) $(CUCFLAGS) $(INC) $(LIB) $^ -o $@
#
#algo.out : \
#      algo.cu
#	CPLUS_INCLUDE_PATH=;\
#  $(CUC) $(CUCFLAGS) $(INC) $(LIB) $^ -o $@
