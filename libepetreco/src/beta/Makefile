override FLAGS +=

CUC = nvcc
CUCFLAGS = -arch=sm_20 $(FLAGS) -Xcompiler -Wall
#CUCFLAGS = -arch=sm_20 -Xcompiler -Wall -DCHORDSCALC_DEBUG

CPPC = g++
CPPCFLAGS = -Wall $(FLAGS)

INC = -I../ -I../../ -I$(HDF5_INC) -I../Ply/
LIB = -L$(HDF5_LIB) -lhdf5 -lhdf5_cpp -lcublas -L../../ -lepetreco

EXECUTABLES := \
      test_backprojection.out \
      test_chunksizes.out \
      test_angular_index_counting.out \
      algo.out



default : \
      $(EXECUTABLES)

clean :
	rm $(EXECUTABLES) \
     ./*.ply \
     ./*.h5


%.out : \
      ./%.cu
	CPLUS_INCLUDE_PATH= ; \
  $(CUC) $(CUCFLAGS) $(INC) $(LIB) $^ -o $@

#real_algo.out : \
#      real_algo.cu
#	CPLUS_INCLUDE_PATH=;\
#  $(CUC) $(CUCFLAGS) $(INC) $(LIB) $^ -o $@
#
#algo.out : \
#      algo.cu
#	CPLUS_INCLUDE_PATH=;\
#  $(CUC) $(CUCFLAGS) $(INC) $(LIB) $^ -o $@


compile_test :
	rm -rf ../compile_test/
	
	# Files that are unknown to svn --> delete_list.txt
	svn st | grep ^? | awk '{print $$2}' > delete_list.txt
	
	# Create compile test directory
	mkdir compile_test
	mv compile_test ..
	
	# Copy all elements from the current directory
	for element in `ls .`; do \
	  if [ -d $$element ]; then \
	    ln -s `pwd`/$$element ../compile_test/$$element; \
	  fi; \
	  if [ -f $$element ]; then \
	    cp $$element ../compile_test/; \
	  fi; \
	done
	
	# Remove files that are unknown to svn
	rm -r `cat delete_list.txt | awk '{print "../compile_test/"$$1}'`
	rm ./delete_list.txt
