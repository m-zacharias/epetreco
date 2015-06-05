#PBS -N test_chunksizes
#PBS -q k20
#PBS -l nodes=1:ppn=8
#PBS -l walltime=5:00:00
#PBS -d .
#PBS -e error.log
#PBS -o output.log
#PBS -M m.zacharias@hzdr.de

PROGRAM='test_chunksizes.out'
OUTDIR='./output/'
TEST_FILES='../../../unversioned_material/epet_POS102.mess.h5'
CHUNKSIZES='3000'
NTHREADRAYS=100

# Create directory, if non-existant
if [ ! -d "$OUTDIR" ] ; then
    mkdir "$OUTDIR"
fi



# Compile program
make $PROGRAM FLAGS=-DWITH_CUDAMATRIX
echo -e "\n\n"

for CHUNKSIZE in $CHUNKSIZES; do
  echo -e "Chunksize: $CHUNKSIZE"
  for TEST_FILE in $TEST_FILES; do
  echo -e "Testfile: $TEST_FILE"
  ./$PROGRAM $TEST_FILE $CHUNKSIZE $RANDOM_SEED $NTHREADRAYS $OUTDIR'/test_chunksize_output'
  echo -e "\n\n"
  done
done 
