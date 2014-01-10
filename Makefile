CPPC = g++
CPPFLAGS = -Wall
CPPO = -c

all : test_Siddon.out

test_Siddon.out : test_Siddon.cpp\
        Siddon.hpp
	$(CPPC) $(CPPFLAGS) $^ -o $@

PlyWriter.o : PlyWriter.cpp\
        PlyWriter.hpp
	$(CPPC) $(CPPFLAGS) $(CPPO) $< -o $@

#Siddon.o : Siddon.hpp
#	$(CPPC) $(CPPFLAGS) $(CPPO) $< -o $@

clean:
	rm  main.out\
      siddon_main.out\
      PlyWriter.o
