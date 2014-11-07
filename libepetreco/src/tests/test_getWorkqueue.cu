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

#include "typedefs.hpp"

void test1(std::string const fn) {
  std::cout << std::endl
            << "-----Test1-----"
            << std::endl;
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
      ML,
      VG, Idx, Idy, Idz,
      MS, Id0z, Id0y, Id1z, Id1y, Ida,
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
  std::cout << std::endl
            << "-----Test2-----"
            << std::endl;
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
  
  nFound = getWorkqueueEntries<
                 val_t,
                 ML,
                 VG, Idx, Idy, Idz,
                 MS, Id0z, Id0y, Id1z, Id1y, Ida,
                 Trafo0, Trafo1>
               (
                 n, wqCnlId, wqVxlId, listId, vxlId, &list, &grid, &setup);
  std::cout << "Found " << nFound << " workqueue entries" << std::endl;
  std::cout << "Workqueue:" << std::endl
            << "----------" << std::endl;
  for(int i=0; i<nFound; i++) {
    std::cout << "  cnlId: " << wqCnlId[i] << ",   vxlId: " << wqVxlId[i]
              << std::endl;
  }
}

void test3( std::string const fn ) {
  std::cout << std::endl
            << "-----Test3-----"
            << std::endl;
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
  
//  int * wqCnlId = new int[n];
  std::vector<int> wqCnlId;
  //  int * wqVxlId = new int[n];
  std::vector<int> wqVxlId;
  int listId(0); int vxlId(0);
  int nFound(0);
  
  nFound = getWorkqueue<
                 val_t,
                 ML,
                 VG, Idx, Idy, Idz,
                 MS, Id0z, Id0y, Id1z, Id1y, Ida,
                 Trafo0, Trafo1>
               (
                wqCnlId, wqVxlId, listId, vxlId, &list, &grid, &setup);
  std::cout << "Found " << nFound << " workqueue entries" << std::endl;
  std::cout << "Workqueue:" << std::endl
            << "----------" << std::endl;
  std::vector<int>::iterator wqCnlIdIt = wqCnlId.begin();
  std::vector<int>::iterator wqVxlIdIt = wqVxlId.begin();
  while((wqCnlIdIt != wqCnlId.end()) && (wqVxlIdIt != wqVxlId.end())) {
    std::cout << "  cnlId: "   << *wqCnlIdIt
              << ",   vxlId: " << *wqVxlIdIt
              << std::endl;
    wqCnlIdIt++;
    wqVxlIdIt++;
  }
}

int main(int argc, char ** argv) {
  int const nargs(2);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected: "
              << nargs << ":" << std::endl
              << "  filename of measurement" << std::endl
              << "  number of workqueue entries to look for" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string const fn(argv[1]);
  int const         n(atoi(argv[2]));
  
  test1(fn);
  test2(fn, n);
  test3(fn);
  
  return 0;
}

