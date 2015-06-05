#PBS -N timetest_2014-07-02
#PBS -q k20f
#PBS -l nodes=1:ppn=8
#PBS -l walltime=5:00:00
#PBS -d .
#PBS -e ./timetest_2014-07-02__/error.log
#PBS -o ./timetest_2014-07-02__/output.log
#PBS -M m.zacharias@hzdr.de

OUTDIR='./timetest_2014-07-02__/'
PROGRAM='test_chunksizes.out'
TEST_FILES='
../beta/test_input/epet_POS102_subset50.mess.h5
../beta/test_input/epet_POS102_subset100.mess.h5
../beta/test_input/epet_POS102_subset200.mess.h5
../beta/test_input/epet_POS102_subset500.mess.h5
../beta/test_input/epet_POS102_subset1000.mess.h5
../beta/test_input/epet_POS102_subset2000.mess.h5
../beta/test_input/epet_POS102_subset5000.mess.h5
../beta/test_input/epet_POS102_subset10000.mess.h5
../beta/test_input/epet_POS102_subset20000.mess.h5
../beta/test_input/epet_POS102_subset50000.mess.h5'

CHUNKSIZES='3000'
NTHREADRAYS=100

# Create directory, if non-existant
if [ ! -d "$OUTDIR" ] ; then
    mkdir "$OUTDIR"
fi



# Compile program
# ALWAYS WITH -DWITH_CUDAMATRIX, OTHERWISE NO BACKPROJECTION!!!
make $PROGRAM FLAGS=-DWITH_CUDAMATRIX

if [ $? -eq 0 ]; then 
  echo -e "\n\n"
  
  for CHUNKSIZE in $CHUNKSIZES; do
    echo -e "Chunksize: $CHUNKSIZE"
    for TEST_FILE in $TEST_FILES; do
    echo -e "Testfile: $TEST_FILE"
  #  echo time ./$PROGRAM $TEST_FILE $CHUNKSIZE $RANDOM_SEED $NTHREADRAYS $OUTDIR$CHUNKSIZE"__"$(basename $TEST_FILE)"_"
    time ./$PROGRAM $TEST_FILE $CHUNKSIZE $RANDOM_SEED $NTHREADRAYS $OUTDIR$CHUNKSIZE"__"$(basename $TEST_FILE)"_"
    echo -e "\n\n"
    done
  done
fi 
