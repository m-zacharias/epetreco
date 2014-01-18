CPPC = g++
CPPCFLAGS = -Wall
#CPPCFLAGS = -Wall -DDEBUG

all : test_List.out\
        test_ListIterator.out

test_List.out : test_List.cpp\
        List.hpp\
        List.tpp
	$(CPPC) $(CPPCFLAGS) $< -o $@

test_ListIterator.out : test_ListIterator.cpp\
        ListIterator.hpp\
        ListIterator.tpp
	$(CPPC) $(CPPCFLAGS) $< -o $@

clean :
	rm ./test_List.out\
     ./test_ListIterator.out
