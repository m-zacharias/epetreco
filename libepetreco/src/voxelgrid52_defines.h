/** @file voxelgrid52_defines.h */



#ifndef VOXELGRID_DEFINES
#define VOXELGRID_DEFINES

#ifdef	__cplusplus
extern "C" {
#endif

#define GRIDNX 52                       /** x dimension of voxel grid */
#define GRIDNY 52                       /** y dimension of voxel grid */
#define GRIDNZ 52                       /** z dimension od voxel grid */
#define GRIDOX -0.026                   /** x origin of voxel grid [m] */
#define GRIDOY -0.026                   /** y origin of voxel grid [m] */
#define GRIDOZ -0.026                   /** z origin of voxel grid [m] */
#define GRIDDX  0.001                   /** x edge length of one voxel [m] */
#define GRIDDY  0.001                   /** y edge length of one voxel [m] */
#define GRIDDZ  0.001                   /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */

#ifdef	__cplusplus
}
#endif

#endif  /* #define VOXELGRID_DEFINES */
