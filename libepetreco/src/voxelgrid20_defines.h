/** @file voxelgrid20_defines.h */
/* Author: malte
 *
 * Created on 4. November 2014, 15:27 */
#ifndef VOXELGRID_DEFINES
#define VOXELGRID_DEFINES

#ifdef	__cplusplus
extern "C" {
#endif

#define GRIDNX 20                       /** x dimension of voxel grid */
#define GRIDNY 20                       /** y dimension of voxel grid */
#define GRIDNZ 20                       /** z dimension od voxel grid */
#define GRIDOX -0.030                   /** x origin of voxel grid [m] */
#define GRIDOY -0.030                   /** y origin of voxel grid [m] */
#define GRIDOZ -0.030                   /** z origin of voxel grid [m] */
#define GRIDDX  0.003                   /** x edge length of one voxel [m] */
#define GRIDDY  0.003                   /** y edge length of one voxel [m] */
#define GRIDDZ  0.003                   /** z edge length of one voxel [m] */
#define VGRIDSIZE (GRIDNX*GRIDNY*GRIDNZ)/** Number of voxel in the voxel grid */

#ifdef	__cplusplus
}
#endif

#endif  /* #define VOXELGRID_DEFINES */
