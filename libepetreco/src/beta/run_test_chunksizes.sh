##TEST_FILES_DIR="../../../unversioned_material/"
#TEST_FILES_DIR="./test_input/"
##TEST_FILES="epet_POS102.mess.h5"
TEST_FILES="epet_POS102_subset15.mess.h5"
##CHUNKSIZES="100 1000 10000 100000"
#CHUNKSIZES="12"
#RANDOM_SEED=1234
#NTHREADRAYS=100
##DEBUGGING_PROGRAM="cuda-memcheck"
#DEBUGGING_PROGRAM=""
#TEST_PROGRAM="test_chunksizes.out"

#TEST_FILES_DIR="../../../unversioned_material/"
TEST_FILES_DIR="./test_input/"
#TEST_FILES="epet_POS102_subset5000.mess.h5"
CHUNKSIZES="100 200"
RANDOM_SEED=1234
NTHREADRAYS=100
DEBUGGING_PROGRAM="cuda-memcheck"
#DEBUGGING_PROGRAM=""
TEST_PROGRAM="test_chunksizes.out"


echo -e "Compile test program...\n-----------------------"
make $TEST_PROGRAM FLAGS=-g FLAGS+=-G FLAGS+=-DWITH_CUDAMATRIX


for CHUNKSIZE in $CHUNKSIZES; do
  echo -e "Chunksize: $CHUNKSIZE\n"
  for TEST_FILE in $TEST_FILES; do
  echo -e "  Testfile: $TEST_FILE\n"
    $DEBUGGING_PROGRAM ./$TEST_PROGRAM $TEST_FILES_DIR$TEST_FILE $CHUNKSIZE $RANDOM_SEED $NTHREADRAYS $TEST_FILE
  done
done 
