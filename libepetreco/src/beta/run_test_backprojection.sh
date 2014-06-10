#TEST_FILES_DIR="../../../unversioned_material/"
TEST_FILES_DIR="./test_input/"
TEST_FILES="epet_POS102_testA.mess.h5 epet_POS102_testB.mess.h5 epet_POS102_testC.mess.h5 epet_POS102_testD.mess.h5 epet_POS102_testE.mess.h5"
CHUNKSIZE=100
RANDOM_SEED=1234
NTHREADRAYS=100

make test_backprojection.out


for TEST_FILE in $TEST_FILES; do
  ./test_backprojection.out $TEST_FILES_DIR$TEST_FILE $CHUNKSIZE $RANDOM_SEED $NTHREADRAYS $TEST_FILE
done 
