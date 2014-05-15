SRC = ./src
TESTS_SRC = ./src/tests
PLY = $(SRC)/Ply

CPPC = g++
override CPPCFLAGS += -Wall
CPPO = -c
INC = -I$(PLY) -I$(SRC) -I$(HDF5_ROOT)/include
LIB = -L$(HDF5_ROOT)/lib -lhdf5 -lhdf5_cpp

CUC = nvcc
CUCFLAGS = -arch=sm_20
CUCINC = -I$(CUDA_ROOT)/include -I$(SRC)
CUCLIB = -L$(CUDA_ROOT)/lib -lcublas

# ##############################################################################
# ### PUT ALL DEFAULT TARGETS HERE
# ##############################################################################
# Define simply expanded variable for all executables. Synchronizes the clean
# rule with the default rule - everything that is compiled by default will also
# be cleaned up.
EXECUTABLES := \
      test_Siddon.out \
      test_02_Siddon.out \
      test_PlyGeometry.out \
      test_PlyBox.out \
      test_PlyWriter.out \
      test_PlyGrid.out \
      test_PlyLine.out \
      test_CudaTransform.out \
      test_CudaTransform2.out \
      test_CudaTransform3.out \
      test_CudaTransform4.out \
      test_CudaTransform5.out \
      test_CudaTransform_DeviceOnly.out \
      test_CudaDeviceOnlyMatrix.out \
      test_H5Reader.out \
      test_H5DensityWriter.out \
      test_ChordsCalc.out \
      test_SMChunkGetter.out \
      test_DevelChannel.out \
      test_ChordsCalc_lowlevel_c.out
#      test_cudaSMLine.out

default : $(EXECUTABLES)

clean:
	rm \
      $(EXECUTABLES) \
      ./libepetreco.a \
      ./*.ply \
      ./*.h5 2>/dev/null; \
  cd ./src/external_library_examples/ && make clean 2>/dev/null


libepetreco.a : \
        PlyGeometry.o \
        CompositePlyGeometry.o \
        PlyWriter.o
	rm -f $@
	ar rcs $@ $^


.INTERMEDIATE : \
        PlyGeometry.o \
        CompositePlyGeometry.o \
        PlyWriter.o



%.out : \
        $(TESTS_SRC)/%.cu
	CPLUS_INCLUDE_PATH= ; \
  $(CUC) $(CUCFLAGS) $(CUCINC) $(CUCLIB) $^ -o $@

test_Ply%.out : \
      $(PLY)/test_Ply%.cpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_%_Siddon.out : \
      $(TESTS_SRC)/test_%_Siddon.cpp \
      $(SRC)/Siddon.hpp \
      $(SRC)/Siddon_helper.hpp \
      libepetreco.a
	$(CPPC) $(CPPFLAGS) $(INC) $^ -o $@

test_%.out : \
      $(TESTS_SRC)/test_%.cpp \
      $(SRC)/%.hpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ $(LIB) -o $@

test_Siddon.out : \
      $(TESTS_SRC)/test_Siddon.cpp\
      $(SRC)/Siddon.hpp
	$(CPPC) $(CPPFLAGS) $(INC) $^ -o $@

bigtest.out : \
      $(TESTS_SRC)/bigtest.cpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@



PlyGeometry.o : \
      $(PLY)/PlyGeometry.cpp \
      $(PLY)/PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

CompositePlyGeometry.o : \
      $(PLY)/CompositePlyGeometry.cpp \
      $(PLY)/CompositePlyGeometry.hpp \
      $(PLY)/PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

PlyWriter.o : \
      $(PLY)/PlyWriter.cpp\
      $(PLY)/PlyWriter.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

