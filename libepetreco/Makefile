SRC = ./src
PLY = $(SRC)/Ply

CPPC = g++
CPPCFLAGS = -Wall
#CPPCFLAGS = -Wall -DDEBUG
CPPO = -c
INC = -I$(PLY) -I$(SRC) -I$(HDF5_ROOT)/include
LIB = -L$(HDF5_ROOT)/lib -lhdf5 -lhdf5_cpp

CUC = nvcc
CUCFLAGS =
CUCINC = -I$(CUDA_ROOT)/include
CUCLIB = -L$(CUDA_ROOT)/lib -lcublas



all : \
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
        cublasExampleMatrixVectorMult.out

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
        



cublasExampleMatrixVectorMult.out : \
        $(SRC)/cublasExampleMatrixVectorMult.cu
	$(CUC) $(CUCFLAGS) $(CUCINC) $(CUCLIB) $^ -o $@


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

test_H5Reader.out : \
        $(SRC)/test_H5Reader.cpp \
        $(SRC)/H5Reader.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) $(LIB) $^ -o $@

test_H5DensityWriter.out : \
        $(SRC)/test_H5DensityWriter.cpp \
        $(SRC)/H5DensityWriter.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) $(LIB) $^ -o $@


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

#Siddon.o : Siddon.hpp
#	$(CPPC) $(CPPFLAGS) $(CPPO) $< -o $@



clean:
	rm \
      ./test_Siddon.out \
      ./test_02_Siddon.out \
      ./test_03_Siddon.out \
      ./test_PlyGeometry.out \
      ./test_PlyWriter.out \
      ./test_PlyGrid.out \
      ./test_PlyBox.out \
      ./test_PlyLine.out \
      ./cublasExampleMatrixVectorMult.out \
      ./test_CudaTransform.out \
      ./test_H5Reader.out \
      ./test_H5DensityWriter.out \
      ./libepetreco.a \
      ./*.ply \
      ./*.h5
