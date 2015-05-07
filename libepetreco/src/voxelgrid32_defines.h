/** @file voxelgrid32_defines.h */
/* Author: malte
 *
 * Created on 20. Februar 2015, 13:54 */
#ifndef VOXELGRID_DEFINES
#define VOXELGRID_DEFINES

#ifdef	__cplusplus
extern "C" {
#endif

#define GRIDNX 32                       /** x dimension of voxel grid */
#define GRIDNY 32                       /** y dimension of voxel grid */
#define GRIDNZ 32                       /** z dimension od voxel grid */
#define GRIDOX -0.030                   /** x origin of voxel grid [m] */
#define GRIDOY -0.030                   /** y origin of voxel grid [m] */
#define GRIDOZ -0.030                   /** z origin of voxel grid [m] */
#define GRIDDX  0.001875                /** x edge length of one voxel [m] */
#define GRIDDY  0.001875                /** y edge length of one voxel [m] */
#define GRIDDZ  0.001875                /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */

#ifdef	__cplusplus
}
#endif

#endif  /* #define VOXELGRID_DEFINES */
