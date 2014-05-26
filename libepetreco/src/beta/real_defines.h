#ifndef DEFINES
#define DEFINES


#define N0Z 13      // 1st detector's number of segments in z
#define N0Y 13      // 1st detector's number of segments in y
#define N1Z 13      // 2nd detector's number of segments in z
#define N1Y 13      // 2nd detector's number of segments in y
#define NA  180     // number of angular positions
#define DA  2.      // angular step
#define POS0X -457. // position of 1st detector's center in x [mm]
#define POS1X  457. // position of 2nd detector's center in x [mm]
#define SEGX 20.    // x edge length of one detector segment [mm]
#define SEGY 4.     // y edge length of one detector segment [mm]
#define SEGZ 4.     // z edge length of one detector segment [mm]
#define NCHANNELS NA*N0Z*N0Y*N1Z*N1Y

#define GRIDNX 10   // x dimension of voxel grid
#define GRIDNY 10   // y dimension of voxel grid
#define GRIDNZ 10   // z dimension od voxel grid
#define GRIDOX -350.// x origin of voxel grid [mm]
#define GRIDOY -350.// y origin of voxel grid [mm]
#define GRIDOZ -350.// z origin of voxel grid [mm]
#define GRIDDX 70.  // x edge length of one voxel [mm]
#define GRIDDY 70.  // y edge length of one voxel [mm]
#define GRIDDZ 70.  // z edge length of one voxel [mm]
#define VGRIDSIZE GRIDNX*GRIDNY*GRIDNZ

#define RANDOM_SEED 1234
#define NTHREADRAYS 100


#endif  // #define DEFINES
