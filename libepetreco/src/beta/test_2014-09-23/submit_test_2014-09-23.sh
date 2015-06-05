#PBS -N test_2014-09-23
#PBS -q k20
#PBS -l nodes=1:ppn=8
#PBS -l walltime=4:00:00
#PBS -d /home/mz6084/epetreco_synced-with-laptop/trunk/libepetreco/src/beta/test_2014-09-23
#PBS -e ./error.log
#PBS -o ./output.log
#PBS -M m.zacharias@hzdr.de

. $HOME/own.modules_script

ROOTDIR='/home/mz6084/epetreco_synced-with-laptop/trunk/libepetreco/src/beta'
SOURCEFILE='test_chunksizes.cu'
PROGRAM='test_chunksizes.out'
LOGFILE="$PBS_O_WORKDIR/log.log"
TEST_FILES="
$ROOTDIR/test_input/epet_POS102_subset100.mess.h5"

CHUNKSIZES='3000'
RANDOM_SEED=1234

# First number is #(rays per thread), second is #(threads per block)
NRAYSPERTHREAD=(1024 512 256 128 64 32 16 8)
NTHREADSPERBLOCK=(1 2 4 8 16 32 64)

echo -n '' > $LOGFILE

# Compile program
# ALWAYS WITH -DWITH_CUDAMATRIX, OTHERWISE NO BACKPROJECTION!!!
cd $ROOTDIR
cp $SOURCEFILE $PBS_O_WORKDIR/
make $PROGRAM FLAGS=-DWITH_CUDAMATRIX &>> $LOGFILE
mv $PROGRAM $PBS_O_WORKDIR/
cd $PBS_O_WORKDIR

if [ $? -eq 0 ]; then 
  echo -e "\n\n" &>> $LOGFILE
  
  for CHUNKSIZE in $CHUNKSIZES; do
    echo -e "Chunksize: $CHUNKSIZE" &>> $LOGFILE
    for TEST_FILE in $TEST_FILES; do
      echo -e "Testfile: $TEST_FILE" &>> $LOGFILE
      for i in `seq 0 6`; do
        echo -e "Rays per thread, threads per block:"\
                "${NRAYSPERTHREAD[$i]}, ${NTHREADSPERBLOCK[$i]}" &>> $LOGFILE
        { time ./$PROGRAM $TEST_FILE $CHUNKSIZE $RANDOM_SEED\
          ${NRAYSPERTHREAD[$i]} ${NTHREADSPERBLOCK[$i]}\
          $PBS_O_WORKDIR/$CHUNKSIZE"__"$(basename $TEST_FILE)"_"; } &>> $LOGFILE
        echo -e "\n\n" >> $LOGFILE
      done
    done
  done
fi 
