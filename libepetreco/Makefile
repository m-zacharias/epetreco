SRC = ./src

CPPC = g++
#CPPFLAGS = -Wall
CPPFLAGS = -Wall -DDEBUG
CPPO = -c
INC = -I$(SRC)/FoundationClasses/

CUC = nvcc
CUCCFLAGS =
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

test_02_Siddon.out : \
        $(SRC)/test_02_Siddon.cpp \
        $(SRC)/Siddon.hpp \
        libepetreco.a
	$(CPPC) $(CPPFLAGS) $(INC) $^ -o $@

test_PlyGeometry.out : \
        $(SRC)/test_PlyGeometry.cpp \
        libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_PlyWriter.out : \
        $(SRC)/test_PlyWriter.cpp \
        libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_PlyGrid.out : \
        $(SRC)/test_PlyGrid.cpp \
        libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_PlyBox.out : \
        $(SRC)/test_PlyBox.cpp \
        libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_PlyLine.out : \
        $(SRC)/test_PlyLine.cpp \
        libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@



PlyGeometry.o : \
        $(SRC)/PlyGeometry.cpp \
        $(SRC)/PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

CompositePlyGeometry.o : \
        $(SRC)/CompositePlyGeometry.cpp \
        $(SRC)/CompositePlyGeometry.hpp \
        $(SRC)/PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

PlyWriter.o : \
        $(SRC)/PlyWriter.cpp\
        $(SRC)/PlyWriter.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

#Siddon.o : Siddon.hpp
#	$(CPPC) $(CPPFLAGS) $(CPPO) $< -o $@



clean:
	rm \
      ./test_Siddon.out \
      ./test_02_Siddon.out \
      ./test_PlyGeometry.out \
      ./test_PlyWriter.out \
      ./test_PlyGrid.out \
      ./test_PlyBox.out \
      ./test_PlyLine.out \
      ./cublasExampleMatrixVectorMult.out \
      ./test_CudaTransform.out \
      ./libepetreco.a
