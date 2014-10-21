/* 
 * File:   test_getWorkqueue.cpp
 * Author: malte
 *
 * Created on 9. Oktober 2014, 17:18
 */

#include "getWorkqueue.hpp"
#include "VoxelGrid.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementSetupLinIndex.hpp"
#include "MeasurementSetupTrafo2CartCoord.hpp"
#include "H5File2DefaultMeasurementList.h"
#include "real_measurementsetup_defines.h"
#include "voxelgrid10_defines.h"
//#include "voxelgrid52_defines.h"
#include <iostream>

typedef float                              val_t;
typedef DefaultVoxelGrid<val_t>            VG;
typedef DefaultMeasurementSetup<val_t>     MS;
typedef DefaultMeasurementSetupId0z<MS>    Id0z;
typedef DefaultMeasurementSetupId0y<MS>    Id0y;
typedef DefaultMeasurementSetupId1z<MS>    Id1z;
typedef DefaultMeasurementSetupId1y<MS>    Id1y;
typedef DefaultMeasurementSetupIda<MS>     Ida;
typedef DefaultMeasurementList<val_t>      ML;
typedef DefaultMeasurementSetupTrafo2CartCoordFirstPixel<val_t, MS>  Trafo0;
typedef DefaultMeasurementSetupTrafo2CartCoordSecndPixel<val_t, MS>  Trafo1;

void test1(std::string const fn) {
  VG grid =
    VG(
      GRIDOX, GRIDOY, GRIDOZ,
      GRIDDX, GRIDDY, GRIDDZ,
      GRIDNX, GRIDNY, GRIDNZ);
    
  MS setup =
    MS(
      POS0X, POS1X,
      NA, N0Z, N0Y, N1Z, N1Y,
      DA, SEGX, SEGY, SEGZ);
  
  ML list =
    H5File2DefaultMeasurementList<val_t>(fn, NA*N0Z*N0Y*N1Z*N1Y);
  
  int wqCnlId; int wqVxlId;
  int listId(0); int vxlId(0);
  int found;
  int nFound(0);
  
  do {
    found = getWorkqueueEntry<
      val_t,
      ML, VG, MS,
      Id0z, Id0y, Id1z, Id1y, Ida,
      Trafo0, Trafo1> (
        &wqCnlId, &wqVxlId, listId, vxlId, &list, &grid, &setup);
    nFound += found;
//    std::cout << "wqCnlId: " << wqCnlId
//              << ", wqVxlId: " << wqVxlId
//              << ", listId: " << listId << std::endl;
  } while(found == 1);
    std::cout << "Looking for all workqueue entries found: " << nFound
              << std::endl;
}

void test2( std::string fn, int const n ) {
  VG grid =
    VG(
      GRIDOX, GRIDOY, GRIDOZ,
      GRIDDX, GRIDDY, GRIDDZ,
      GRIDNX, GRIDNY, GRIDNZ);
    
  MS setup =
    MS(
      POS0X, POS1X,
      NA, N0Z, N0Y, N1Z, N1Y,
      DA, SEGX, SEGY, SEGZ);
  
  ML list =
    H5File2DefaultMeasurementList<val_t>(fn, NA*N0Z*N0Y*N1Z*N1Y);
  
  int * wqCnlId = new int[n];
  int * wqVxlId = new int[n];
  int listId(0); int vxlId(0);
  int nFound;
  
  nFound = getWorkqueueEntries<val_t,
                               ML, VG, MS,
                               Id0z, Id0y, Id1z, Id1y, Ida,
                               Trafo0, Trafo1> (
             n, wqCnlId, wqVxlId, listId, vxlId, &list, &grid, &setup);
  std::cout << "Found " << nFound << " workqueue entries" << std::endl;
  std::cout << "Workqueue:" << std::endl
            << "----------" << std::endl;
  for(int i=0; i<n; i++) {
    std::cout << "  cnlId: " << wqCnlId[i] << ",   vxlId: " << wqVxlId[i]
              << std::endl;
  }
}

int main() {
  std::string fn;
  int n;
  std::cout << "Enter filename of measurement data: ";
  std::cin >> fn;
  std::cout << "Look for how many workqueue entries?: ";
  std::cin >> n;
  
  test1(fn);
  test2(fn, n);
  
  return 0;
}

