/** @file typedefs.hpp */
/* 
 * Author: malte
 *
 * Created on 24. Oktober 2014, 11:41
 */

#ifndef TYPEDEFS_HPP
#define	TYPEDEFS_HPP

#include "VoxelGrid.hpp"
#include "VoxelGridLinIndex.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementSetupLinIndex.hpp"
#include "MeasurementSetupTrafo2CartCoord.hpp"
#include "MeasurementList.hpp"

typedef float                              val_t;
typedef DefaultVoxelGrid<val_t>            VG;
typedef DefaultVoxelGridIdx<VG>            Idx;
typedef DefaultVoxelGridIdy<VG>            Idy;
typedef DefaultVoxelGridIdz<VG>            Idz;
typedef DefaultMeasurementSetup<val_t>     MS;
typedef DefaultMeasurementSetupId0z<MS>    Id0z;
typedef DefaultMeasurementSetupId0y<MS>    Id0y;
typedef DefaultMeasurementSetupId1z<MS>    Id1z;
typedef DefaultMeasurementSetupId1y<MS>    Id1y;
typedef DefaultMeasurementSetupIda<MS>     Ida;
typedef DefaultMeasurementList<val_t>      ML;
typedef DefaultMeasurementSetupTrafo2CartCoordFirstPixel<val_t, MS>  Trafo0;
typedef DefaultMeasurementSetupTrafo2CartCoordSecndPixel<val_t, MS>  Trafo1;

#endif	/* TYPEDEFS_HPP */
