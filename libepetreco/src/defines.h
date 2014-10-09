#ifndef DEFINES
#define DEFINES


#define N0Z 5       // 1st detector's number of segments in z
#define N0Y 1       // 1st detector's number of segments in y
#define N1Z 4       // 2nd detector's number of segments in z
#define N1Y 1       // 2nd detector's number of segments in y
#define NA  1       // number of angular positions
#define DA  5.      // angular step
#define POS0X -3.5  // position of 1st detector's center in x
#define POS1X  3.5  // position of 2nd detector's center in x
#define SEGX 1.     // x edge length of one detector segment
#define SEGY 1.     // y edge length of one detector segment
#define SEGZ 1.     // z edge length of one detector segment
#define NCHANNELS NA*N0Z*N0Y*N1Z*N1Y

#define GRIDNX 3    // x dimension of voxel grid
#define GRIDNY 1    // y dimension of voxel grid
#define GRIDNZ 4    // z dimension od voxel grid
#define VGRIDSIZE GRIDNX*GRIDNY*GRIDNZ

#define NTHREADRAYS 20
#define NBLOCKS NCHANNELS
#define NTHREADS 1

//#define DEBUG 1
#define PRINT_KERNEL 0

#define RANDOM_SEED 1234


#endif  // #define DEFINES
