CPPC = g++
CPPFLAGS = -Wall
#CPPFLAGS = -Wall -DDEBUG
CPPO = -c
INC = -I./FoundationClasses/

all : \
        test_Siddon.out\
        test2_Siddon.out\
        test_PlyGeometry.out\
        test_PlyBox.out\
        test_PlyWriter.out\
        test_PlyGrid.out



test_Siddon.out : \
        test_Siddon.cpp\
        Siddon.hpp
	$(CPPC) $(CPPFLAGS) $(INC) $^ -o $@

test2_Siddon.out : \
        test2_Siddon.cpp\
        Siddon.hpp
	$(CPPC) $(CPPFLAGS) $(INC) $^ -o $@

test_PlyGeometry.out : \
        test_PlyGeometry.cpp\
        Vertex.o\
        PlyGeometry.o\
        PlyRectangle.o\
        CompositePlyGeometry.o
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_PlyWriter.out : \
        test_PlyWriter.cpp\
        Vertex.o\
        PlyGeometry.o\
        PlyRectangle.o\
        CompositePlyGeometry.o\
        PlyWriter.o
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_PlyGrid.out : \
        test_PlyGrid.cpp\
        Vertex.o\
        PlyGeometry.o\
        PlyRectangle.o\
        CompositePlyGeometry.o\
        PlyGrid.o\
        PlyWriter.o
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_PlyBox.out : \
        test_PlyBox.cpp\
        Vertex.o\
        PlyGeometry.o\
        CompositePlyGeometry.o\
        PlyBox.o\
        PlyWriter.o
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@

test_PlyLine.out : \
        test_PlyLine.cpp\
        PlyLine.o\
        Vertex.o\
        PlyGeometry.o\
        PlyWriter.o
	$(CPPC) $(CPPCFLAGS) $(INC) $^ -o $@



PlyGeometry.o : \
        PlyGeometry.cpp\
        PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

CompositePlyGeometry.o : \
        CompositePlyGeometry.cpp\
        CompositePlyGeometry.hpp\
        PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

PlyRectangle.o : \
        PlyRectangle.cpp\
        PlyRectangle.hpp\
        PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

PlyBox.o : \
        PlyBox.cpp\
        PlyBox.hpp\
        PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

PlyLine.o : \
        PlyLine.cpp\
        PlyLine.hpp\
        PlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

PlyGrid.o : \
        PlyGrid.cpp\
        PlyGrid.hpp\
        PlyGeometry.hpp\
        PlyRectangle.hpp\
        CompositePlyGeometry.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

PlyWriter.o : \
        PlyWriter.cpp\
        PlyWriter.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

Vertex.o : \
        Vertex.cpp\
        Vertex.hpp
	$(CPPC) $(CPPCFLAGS) $(INC) -c $< -o $@

#Siddon.o : Siddon.hpp
#	$(CPPC) $(CPPFLAGS) $(CPPO) $< -o $@



clean:
	rm\
      ./test_Siddon.out\
      ./test2_Siddon.out\
      ./test_PlyGeometry.out\
      ./test_PlyWriter.out\
      ./test_PlyGrid.out\
      ./test_PlyBox.out\
      ./test_PlyLine.out\
      ./Vertex.o\
      ./PlyGeometry.o\
      ./CompositePlyGeometry.o\
      ./PlyRectangle.o\
      ./PlyBox.o\
      ./PlyLine.o\
      ./PlyGrid.o\
      ./PlyWriter.o
