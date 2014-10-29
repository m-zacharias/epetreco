/* 
 * File:   getWorkqueue.hpp
 * Author: malte
 *
 * Created on 9. Oktober 2014, 14:53
 */

#ifndef GETWORKQUEUE_HPP
#define	GETWORKQUEUE_HPP

#include <cmath>
#include "VoxelGrid.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementList.hpp"
#include "distancePointLine.h"

template<typename T,
         typename ConcreteList,
         typename ConcreteVG,
         typename ConcreteMS,
         typename ConcreteMSid0z,
         typename ConcreteMSid0y,
         typename ConcreteMSid1z,
         typename ConcreteMSid1y,
         typename ConcreteMSida,
         typename ConcreteMSTrafo2CartCoordFirstPixel,
         typename ConcreteMSTrafo2CartCoordSecndPixel>
int getWorkqueueEntry(int * const returnCnlId, int * const returnVxlId,
                      int & listId, int & vxlId,
                      ConcreteList const * const list,
                      ConcreteVG const * const grid,
                      ConcreteMS const * const setup) {
  // Create functors
  ConcreteMSid0z f_id0z;
  ConcreteMSid0y f_id0y;
  ConcreteMSid1z f_id1z;
  ConcreteMSid1y f_id1y;
  ConcreteMSida  f_ida;
  ConcreteMSTrafo2CartCoordFirstPixel trafo0;
  ConcreteMSTrafo2CartCoordSecndPixel trafo1;
  
  // Relative 3d center coordinates
  T center[3] = {0.5, 0.5, 0.5};
  
  // Sum of radii
  T vxlEdges[3] = {grid->griddx(), grid->griddy(), grid->griddz()};
  T pxlEdges[3] = {setup->segx(), setup->segy(), setup->segz()};
  T sumRadii = 0.5*(absolute(vxlEdges)+absolute(pxlEdges));
  
  // Scan (list, grid) for next candidate pair (cnlId, vxlId)
  while(listId < list->size()) {
    // Get channel id of current list entry
    int cnlId = list->cnlId(listId);
    
    
    // Get channel pixel centers, voxel center
    T pix0Center[3];
    trafo0(pix0Center, center, f_id0z(cnlId, setup), f_id0y(cnlId, setup), f_ida(cnlId, setup), setup);
    
    T pix1Center[3];
    trafo1(pix1Center, center, f_id1z(cnlId, setup), f_id1y(cnlId, setup), f_ida(cnlId, setup), setup);
    
    T vxlCenter[3];
    int sepVxlId[3];
    grid->sepVoxelId(sepVxlId, vxlId);
    vxlCenter[0] = grid->gridox() + (sepVxlId[0]+center[0])*grid->griddx();
    vxlCenter[1] = grid->gridoy() + (sepVxlId[1]+center[1])*grid->griddy();
    vxlCenter[2] = grid->gridoz() + (sepVxlId[2]+center[2])*grid->griddz();
    
    
    // If is candidate ...
//    if(true) {
    if(distance(pix0Center, pix1Center, vxlCenter)<sumRadii) {
      // ... save cnlId, vxlId; inkrement ids; return 1
      *returnCnlId = cnlId;
      *returnVxlId = vxlId;
      
      if(vxlId < (grid->gridnx()*grid->gridny()*grid->gridnz())-1) {
        vxlId++;
      } else if(vxlId == (grid->gridnx()*grid->gridny()*grid->gridnz())-1) {
        vxlId = 0;
        listId++;
      } else {
        std::cerr << "Invalid voxel index encountered during function "
                  << "'getWorkqueueEntry(...)': " << vxlId << std::endl;
        exit(EXIT_FAILURE);
      }
      
      // return: 1 candidate found
      return 1;
    }
      
    // ... else: inkrement ids
    if(vxlId < (grid->gridnx()*grid->gridny()*grid->gridnz())-1) {
      vxlId++;
    } else if(vxlId == (grid->gridnx()*grid->gridny()*grid->gridnz())-1) {
      vxlId = 0;
      listId++;
    } else {
      std::cerr << "Invalid voxel index encountered during function "
                << "'getWorkqueueEntry(...)': " << vxlId << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  
  // return: 0 candidates found
  return 0;
}

template<typename T,
         typename ConcreteList,
         typename ConcreteVG,
         typename ConcreteMS,
         typename ConcreteMSid0z,
         typename ConcreteMSid0y,
         typename ConcreteMSid1z,
         typename ConcreteMSid1y,
         typename ConcreteMSida,
         typename ConcreteMSTrafo2CartCoordFirstPixel,
         typename ConcreteMSTrafo2CartCoordSecndPixel>
int getWorkqueueEntries( int const n,
                      int * const returnCnlId, int * const returnVxlId,
                      int & listId, int & vxlId,
                      ConcreteList const * const list,
                      ConcreteVG const * const grid,
                      ConcreteMS const * const setup ) {
  int nFound(0);
  int found(0);
  do {
    found = getWorkqueueEntry<T,
                              ConcreteList,
                              ConcreteVG,
                              ConcreteMS,
                              ConcreteMSid0z,
                              ConcreteMSid0y,
                              ConcreteMSid1z,
                              ConcreteMSid1y,
                              ConcreteMSida,
                              ConcreteMSTrafo2CartCoordFirstPixel,
                              ConcreteMSTrafo2CartCoordSecndPixel> (
              &returnCnlId[nFound], &returnVxlId[nFound],
              listId, vxlId,
              list, grid, setup);
    nFound += found;
  } while((nFound<n) && (found==1));
  return nFound;
}

#endif	/* GETWORKQUEUE_HPP */
