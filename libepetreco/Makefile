SRC = ./src
PLY = $(SRC)/Ply

CPPC = g++
override CPPCFLAGS += -Wall
CPPO = -c
INC = -I$(PLY) -I$(SRC) -I$(HDF5_ROOT)/include
LIB = -L$(HDF5_ROOT)/lib -lhdf5 -lhdf5_cpp

CUC = nvcc
CUCFLAGS =
CUCINC = -I$(CUDA_ROOT)/include
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
      test_H5Reader.out \
      test_H5DensityWriter.out \
      test_ChordsCalc.out \
      test_SMChunkGetter.out \
      test_DevelChannel.out \
      test_CudaTransform.out \
      cublasExampleMatrixVectorMult.out \
      cublasExampleMatrixVectorMult2.out \
      cublasxtExampleMatrixVectorMult.out





default : $(EXECUTABLES)


clean:
	rm \
      $(EXECUTABLES) \
      ./libepetreco.a \
      ./*.ply \
      ./*.h5


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
        $(SRC)/%.cu
	$(CUC) $(CUCFLAGS) $(CUCINC) $(CUCLIB) $^ -o $@

#cublasExampleMatrixVectorMult2.out : \
#        $(SRC)/cublasExampleMatrixVectorMult2.cu
#	$(CUC) $(CUCFLAGS) $(CUCINC) $(CUCLIB) $^ -o $@


test_CudaTransform.out :\
        $(SRC)/test_CudaTransform.cu\
        $(SRC)/CudaMatrix.hpp\
        $(SRC)/CudaMatrix.tpp\
        $(SRC)/CudaVector.hpp\
        $(SRC)/CudaVector.tpp\
        $(SRC)/CudaTransform.hpp\
        $(SRC)/CudaTransform.tpp\
        $(SRC)/conversion.hpp\
        $(SRC)/cublas_gemv.hpp\
        $(SRC)/cublas_gemv.tpp
	$(CUC) $(CUCFLAGS) $(CUCINC) $(CUCLIB) $< -o $@


test_Siddon.out : \
        $(SRC)/test_Siddon.cpp\
        $(SRC)/Siddon.hpp
	$(CPPC) $(CPPFLAGS) $(INC) $^ -o $@

test_%_Siddon.out : \
        $(SRC)/test_%_Siddon.cpp \
        $(SRC)/Siddon.hpp \
        $(SRC)/Siddon_helper.hpp \
        libepetreco.a
	$(CPPC) $(CPPFLAGS) $(INC) $^ -o $@

test_Ply%.out : \
        $(PLY)/test_Ply%.cpp \
        libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

bigtest.out : \
        $(SRC)/bigtest.cpp \
        libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_%.out : \
      $(SRC)/test_%.cpp \
      $(SRC)/%.hpp \
      libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ $(LIB) -o $@


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

