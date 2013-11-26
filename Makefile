CPPC = g++
CPPFLAGS = -Wall
CPPO = -c

all : main.out

main.out : main.cpp\
        PlyWriter.o
	$(CPPC) $(CPPFLAGS) $^ -o $@

siddon_main.out : siddon_main.cpp\
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
