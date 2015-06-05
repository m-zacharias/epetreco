#PBS -N timetest_2014-07-09
#PBS -q k20
#PBS -l nodes=1:ppn=8
#PBS -l walltime=48:00:00
#PBS -d /home/mz6084/svn/epetreco/trunk/libepetreco/src/beta/timetest_2014-07-09
#PBS -e ./error.log
#PBS -o ./output.log
#PBS -M m.zacharias@hzdr.de

ROOTDIR='/home/mz6084/svn/epetreco/trunk/libepetreco/src/beta'
PROGRAM='test_chunksizes.out'
LOGFILE="$PBS_O_WORKDIR/log.log"
TEST_FILES="
$ROOTDIR/test_input/epet_POS102_subset50.mess.h5
$ROOTDIR/test_input/epet_POS102_subset100.mess.h5
$ROOTDIR/test_input/epet_POS102_subset200.mess.h5
$ROOTDIR/test_input/epet_POS102_subset500.mess.h5
$ROOTDIR/test_input/epet_POS102_subset1000.mess.h5
$ROOTDIR/test_input/epet_POS102_subset2000.mess.h5
$ROOTDIR/test_input/epet_POS102_subset5000.mess.h5
$ROOTDIR/test_input/epet_POS102_subset10000.mess.h5
$ROOTDIR/test_input/epet_POS102_subset20000.mess.h5
$ROOTDIR/test_input/epet_POS102_subset50000.mess.h5
$ROOTDIR/test_input/epet_POS102.mess.h5"

CHUNKSIZES='3000'
RANDOM_SEED=1234
NTHREADRAYS=100


echo -n '' > $LOGFILE

# Compile program
# ALWAYS WITH -DWITH_CUDAMATRIX, OTHERWISE NO BACKPROJECTION!!!
cd $ROOTDIR
make $PROGRAM FLAGS=-DWITH_CUDAMATRIX &>> $LOGFILE
mv $PROGRAM $PBS_O_WORKDIR/
cd $PBS_O_WORKDIR

if [ $? -eq 0 ]; then 
  echo -e "\n\n" &>> $LOGFILE
  
  for CHUNKSIZE in $CHUNKSIZES; do
    echo -e "Chunksize: $CHUNKSIZE" &>> $LOGFILE
    for TEST_FILE in $TEST_FILES; do
    echo -e "Testfile: $TEST_FILE" &>> $LOGFILE
    # echo time ./$PROGRAM $TEST_FILE $CHUNKSIZE $RANDOM_SEED $NTHREADRAYS $DIR/$CHUNKSIZE"__"$(basename $TEST_FILE)"_"
    { time ./$PROGRAM $TEST_FILE $CHUNKSIZE $RANDOM_SEED $NTHREADRAYS $PBS_O_WORKDIR/$CHUNKSIZE"__"$(basename $TEST_FILE)"_"; } &>> $LOGFILE
    echo -e "\n\n" >> $LOGFILE
    done
  done
fi 
