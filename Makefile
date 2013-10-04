CPPC = g++
CPPFLAGS = -Wall -DDEBUG
CPPO = -c

all : main.out

main.out : main.cpp\
        PlyWriter.o
	$(CPPC) $(CPPFLAGS) $^ -o $@

PlyWriter.o : PlyWriter.cpp\
        PlyWriter.hpp
	$(CPPC) $(CPPFLAGS) $(CPPO) $< -o $@

clean:
	rm main.out PlyWriter.o
