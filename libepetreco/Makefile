SRC = ./src
PLY = $(SRC)/Ply

CPPC = g++
#CPPFLAGS = -Wall
CPPFLAGS = -Wall -DDEBUG
CPPO = -c
#INC = -I$(SRC)/FoundationClasses/ -I$(PLY) -I$(SRC)
INC = -I$(PLY) -I$(SRC)

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

test_%_Siddon.out : \
        $(SRC)/test_%_Siddon.cpp \
        $(SRC)/Siddon.hpp \
        $(SRC)/Siddon_helper.hpp \
        libepetreco.a
	$(CPPC) $(CPPFLAGS) $(INC) $^ -o $@

#test_PlyGeometry.out : \
#        $(SRC)/test_PlyGeometry.cpp \
#        libepetreco.a
#	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@
#
#test_PlyWriter.out : \
#        $(SRC)/test_PlyWriter.cpp \
#        libepetreco.a
#	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@
#
#test_PlyGrid.out : \
#        $(SRC)/test_PlyGrid.cpp \
#        libepetreco.a
#	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@
#
#test_PlyBox.out : \
#        $(SRC)/test_PlyBox.cpp \
#        libepetreco.a
#	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@
#
#test_PlyLine.out : \
#        $(SRC)/test_PlyLine.cpp \
#        libepetreco.a
#	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_Ply%.out : \
        $(PLY)/test_Ply%.cpp \
        libepetreco.a
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

bigtest.out : \
        $(SRC)/bigtest.cpp \
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
      ./libepetreco.a \
      ./*.ply
