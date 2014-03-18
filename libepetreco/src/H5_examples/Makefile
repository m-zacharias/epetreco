CPPC = g++
CPPCFLAGS = -Wall
INC = -I$(HDF5_ROOT)/include
LIB = -L$(HDF5_ROOT)/lib -lhdf5 -lhdf5_cpp

default :


all : \
      example1_H5read.out \
      example2_H5read.out \
      example_H5write.out \
      example_H5CompoundType.out


example%.out : \
      example%.cpp
	$(CPPC) $(CPPCFLAGS) $(INC) $(LIB) $< -o $@


clean :
	rm ./*.out \
     ./*.h5
