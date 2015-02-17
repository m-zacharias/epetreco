/** @file voxelgrid10_defines.h */



#ifndef VOXELGRID_DEFINES
#define VOXELGRID_DEFINES

#ifdef	__cplusplus
extern "C" {
#endif

#define GRIDNX 10                       /** x dimension of voxel grid */
#define GRIDNY 10                       /** y dimension of voxel grid */
#define GRIDNZ 10                       /** z dimension od voxel grid */
#define GRIDOX -0.350                   /** x origin of voxel grid [m] */
#define GRIDOY -0.350                   /** y origin of voxel grid [m] */
#define GRIDOZ -0.350                   /** z origin of voxel grid [m] */
#define GRIDDX  0.070                   /** x edge length of one voxel [m] */
#define GRIDDY  0.070                   /** y edge length of one voxel [m] */
#define GRIDDZ  0.070                   /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */

#ifdef	__cplusplus
}
#endif

#endif  /* #define VOXELGRID_DEFINES */
