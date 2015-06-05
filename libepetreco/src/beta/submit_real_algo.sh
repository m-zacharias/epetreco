#PBS -N real_algo
#PBS -q k20
#PBS -l nodes=1:ppn=8
#PBS -l walltime=5:00:00
#PBS -d .
#PBS -e error.log
#PBS -o output.log
#PBS -M m.zacharias@hzdr.de

PROGRAM='real_algo.out'
OUTDIR='./output/'
MEASUREMENT_DATA='../../../unversioned_material/epet_POS102.mess.h5'

# Create directory, if non-existant
if [ ! -d "$OUTDIR" ] ; then
    mkdir "$OUTDIR"
fi



# Compile program
make $PROGRAM

# Program execution
./$PROGRAM $MEASUREMENT_DATA
