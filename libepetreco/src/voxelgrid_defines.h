/** @file voxelgrid_defines.h
 * Author: malte
 *
 * Created on 21. April 2015, 14:02 */

#ifndef VOXELGRID_DEFINES_H
#define	VOXELGRID_DEFINES_H

#ifdef	__cplusplus
extern "C" {
#endif

  
#ifdef GRID64
#include "voxelgrid64_defines.h"

#else
#ifdef GRID52
#include "voxelgrid52_defines.h"
  
#else
#ifdef GRID32
#include "voxelgrid32_defines.h"
  
#else
#ifdef GRID20
#include "voxelgrid20_defines.h"

#else
#ifdef GRID10
#include "voxelgrid10_defines.h"

#endif /* GRID10 */
#endif /* GRID20 */
#endif /* GRID32 */
#endif /* GRID52 */
#endif /* GRID64 */


#ifdef	__cplusplus
}
#endif

#endif	/* VOXELGRID_DEFINES_H */

