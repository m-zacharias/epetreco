CPPC = g++
CPPFLAGS = -Wall
CPPO = -c

all : test_Siddon.out\
        test_PlyOutput.out

test_Siddon.out : test_Siddon.cpp\
        Siddon.hpp
	$(CPPC) $(CPPFLAGS) $^ -o $@

test_PlyOutput.out : test_PlyOutput.cpp\
        PlyOutput.o
	$(CPPC) $(CPPFLAGS) $^ -o $@

PlyOutput.o : PlyOutput.cpp\
        PlyOutput.hpp
	$(CPPC) $(CPPFLAGS) $(CPPO) $< -o $@

#Siddon.o : Siddon.hpp
#	$(CPPC) $(CPPFLAGS) $(CPPO) $< -o $@

clean:
	rm\
      test_Siddon.out\
      test_PlyOutput.out\
      PlyOutput.o
