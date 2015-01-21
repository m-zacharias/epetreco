#ifndef GETSYSTEMMATRIXDEVICEONLY_CU
#define GETSYSTEMMATRIXDEVICEONLY_CU

#include "VoxelGrid.hpp"
#include "VoxelGridLinIndex.hpp"
#include "MeasurementSetup.hpp"
#include "ChordsCalc_lowlevel.hpp"
#include <curand_kernel.h>
#include "device_constant_memory.hpp"

#include <cmath>
#include <vector>
#include "VoxelGrid.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementList.hpp"
#include "distancePointLine.h"

#define RANDOM_SEED 1251
#define TPB 256

template<
      typename T
    , typename ConcreteVG
    , typename ConcreteVGidx
    , typename ConcreteVGidy
    , typename ConcreteVGidz
    , typename ConcreteMS
    , typename ConcreteMSid0z
    , typename ConcreteMSid0y
    , typename ConcreteMSid1z
    , typename ConcreteMSid1y
    , typename ConcreteMSida
    , typename ConcreteMSTrafo2CartCoordFirstPixel
    , typename ConcreteMSTrafo2CartCoordSecndPixel>
__device__
bool test(
      int const cnlId, int const vxlId) {
  
  // Create functors
  ConcreteVGidx  f_idx;
  ConcreteVGidy  f_idy;
  ConcreteVGidz  f_idz;
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
  T vxlEdges[3] = {grid_const.griddx(), grid_const.griddy(), grid_const.griddz()};
  T pxlEdges[3] = {setup_const.segx(), setup_const.segy(), setup_const.segz()};
  T sumRadii = 0.5*(absolute(vxlEdges)+absolute(pxlEdges));
  
  // Get channel pixel centers, voxel center
  T pix0Center[3];
  trafo0(pix0Center, center, f_id0z(cnlId, &setup_const), f_id0y(cnlId, &setup_const), f_ida(cnlId, &setup_const), &setup_const);

  T pix1Center[3];
  trafo1(pix1Center, center, f_id1z(cnlId, &setup_const), f_id1y(cnlId, &setup_const), f_ida(cnlId, &setup_const), &setup_const);

  T vxlCenter[3];
  vxlCenter[0] = grid_const.gridox() + (f_idx(vxlId, &grid_const)+center[0])*grid_const.griddx();
  vxlCenter[1] = grid_const.gridoy() + (f_idy(vxlId, &grid_const)+center[1])*grid_const.griddy();
  vxlCenter[2] = grid_const.gridoz() + (f_idz(vxlId, &grid_const)+center[2])*grid_const.griddz();

  
  if(distance(pix0Center, pix1Center, vxlCenter)<sumRadii) {
    return true;
  }
  return false;
}



template<
      typename T >
__host__ __device__
T intersectionLength(
      T const * const vxlCoord,
      T const * const rayCoord ) {
 
  // Which planes are intersected?  If non at all:  Return zero
  bool sects[3];
  for(int dim=0; dim<3; dim++)
    sects[dim] = rayCoord[dim] != rayCoord[3+dim];
  if(!(sects[0]||sects[1]||sects[2]))
    return 0;
  
  // Get min, max intersection parameters for each dim
  T aDimMin[3];
  T aDimMax[3];
  T temp;
  for(int dim=0; dim<3; dim++) {
    if(sects[dim]) {
      aDimMin[dim] = (vxlCoord[  dim]-rayCoord[dim])
                    /(rayCoord[3+dim]-rayCoord[dim]);
      aDimMax[dim] = (vxlCoord[3+dim]-rayCoord[dim])
                    /(rayCoord[3+dim]-rayCoord[dim]);
      if(aDimMax[dim]<aDimMin[dim]) {
        temp         = aDimMin[dim];
        aDimMin[dim] = aDimMax[dim];
        aDimMax[dim] = temp;
      }
    }
  }
  
  // Get entry and exit points
  T aMin, aMax;
  bool aMinGood, aMaxGood;
  MaxFunctor<3>()(&aMin, &aMinGood, aDimMin, sects);
  MinFunctor<3>()(&aMax, &aMaxGood, aDimMax, sects);
  
  // Really intersects?
  if(!(aMin<aMax) || (!aMinGood) || (!aMaxGood))
    return 0;
  
  return (aMax-aMin) * sqrt(  (rayCoord[3+0]-rayCoord[0]) * (rayCoord[3+0]-rayCoord[0])
                             +(rayCoord[3+1]-rayCoord[1]) * (rayCoord[3+1]-rayCoord[1])
                             +(rayCoord[3+2]-rayCoord[2]) * (rayCoord[3+2]-rayCoord[2]) );
}



template<
      typename T
    , typename ConcreteVG
    , typename ConcreteVGidx
    , typename ConcreteVGidy
    , typename ConcreteVGidz
    , typename ConcreteMS
    , typename ConcreteMSid0z
    , typename ConcreteMSid0y
    , typename ConcreteMSid1z
    , typename ConcreteMSid1y
    , typename ConcreteMSida  
    , typename ConcreteMSTrafo2CartCoordFirstPixel
    , typename ConcreteMSTrafo2CartCoordSecndPixel >
__device__
T calcSme(
      int const cnl,
      int const vxl) {
  
  // Create functors
  ConcreteVGidx  f_idx;
  ConcreteVGidy  f_idy;
  ConcreteVGidz  f_idz;
  ConcreteMSid0z f_id0z;
  ConcreteMSid0y f_id0y;
  ConcreteMSid1z f_id1z;
  ConcreteMSid1y f_id1y;
  ConcreteMSida  f_ida;
  ConcreteMSTrafo2CartCoordFirstPixel trafo0;
  ConcreteMSTrafo2CartCoordSecndPixel trafo1;

  // Calculate voxel coordinates
  T vxlCoord[6];
  int sepVxlId[3];
  sepVxlId[0] = f_idx(vxl, &grid_const);
  sepVxlId[1] = f_idy(vxl, &grid_const);
  sepVxlId[2] = f_idz(vxl, &grid_const);

  vxlCoord[0] = grid_const.gridox() +  sepVxlId[0]   *(grid_const.griddx());
  vxlCoord[1] = grid_const.gridoy() +  sepVxlId[1]   *(grid_const.griddy());
  vxlCoord[2] = grid_const.gridoz() +  sepVxlId[2]   *(grid_const.griddz());
  vxlCoord[3] = grid_const.gridox() + (sepVxlId[0]+1)*(grid_const.griddx());
  vxlCoord[4] = grid_const.gridoy() + (sepVxlId[1]+1)*(grid_const.griddy());
  vxlCoord[5] = grid_const.gridoz() + (sepVxlId[2]+1)*(grid_const.griddz());

  // Initialize random number generator
  curandState rndState;
  curand_init(RANDOM_SEED, cnl, 0, &rndState);

  // Matrix element
  T a(0);

  // For rays ...
  for(int idRay=0; idRay<nrays_const; idRay++) {
    // ... Get 6 randoms for ray ...
    T rnd[6];
    rnd[0] = curand_uniform(&rndState);
    rnd[1] = curand_uniform(&rndState);
    rnd[2] = curand_uniform(&rndState);
    rnd[3] = curand_uniform(&rndState);
    rnd[4] = curand_uniform(&rndState);
    rnd[5] = curand_uniform(&rndState);

    // ... Calculate ray coordinates ...
    T rayCoord[6];
    int id0z = f_id0z(cnl, &setup_const);
    int id0y = f_id0y(cnl, &setup_const);
    int id1z = f_id1z(cnl, &setup_const);
    int id1y = f_id1y(cnl, &setup_const);
    int ida  = f_ida(cnl, &setup_const);

    trafo0(&rayCoord[0], &rnd[0], id0z, id0y, ida, &setup_const);
    trafo1(&rayCoord[3], &rnd[3], id1z, id1y, ida, &setup_const);

    // ... Calculate & add intersection length
    a += intersectionLength(vxlCoord, rayCoord);
  }

  // Divide by number of rays
  a /= nrays_const;

  return a;
}



template< typename VG >
struct GetId2vxlId {
  int const _gridDim;
  
  __host__ __device__
  GetId2vxlId( VG & vg )
  : _gridDim(vg.gridnx()*vg.gridny()*vg.gridnz()) {}
  
  __host__ __device__
  int operator()( int const getId ) const {
    return getId % _gridDim;
  }
};



template< typename VG >
struct GetId2listId {
  int const _gridDim;
  
  __host__ __device__
  GetId2listId( VG & vg )
  : _gridDim(vg.gridnx()*vg.gridny()*vg.gridnz()) {}

  __host__ __device__
  int operator()( int const getId ) const {
    return getId / _gridDim;
  }  
};



template<
      typename ConcreteVG
>
struct IsInRange {
  int const _gridDim;
  int const _listSize;
  
  __device__
  IsInRange( VG & vg, int const mlSize )
  : _gridDim(vg.gridnx()*vg.gridny()*vg.gridnz()),
    _listSize(mlSize) {}
  
  __device__
  bool operator()( int const getId ) const {
    return getId < ((_gridDim * _listSize) + blockDim.x -1);
  }
};



template< typename ConcreteVG >
struct GetIsLegal {
  int const _gridDim;
  int const _listSize;
  
  __host__ __device__
  GetIsLegal( VG vg, int const mlSize )
  : _gridDim(vg.gridnx()*vg.gridny()*vg.gridnz()),
    _listSize(mlSize) {}
  
  __host__ __device__
  bool operator()( int const vxlId, int const listId ) const {
    return (listId<_listSize) && (vxlId<_gridDim);
  }
};



template<
      typename T
    , typename ConcreteVG
    , typename ConcreteVGidx
    , typename ConcreteVGidy
    , typename ConcreteVGidz
    , typename ConcreteMS
    , typename ConcreteMSid0z
    , typename ConcreteMSid0y
    , typename ConcreteMSid1z
    , typename ConcreteMSid1y
    , typename ConcreteMSida  
    , typename ConcreteMSTrafo2CartCoordFirstPixel
    , typename ConcreteMSTrafo2CartCoordSecndPixel
>
__global__
void getSystemMatrix(
      T * const   sme_devi,   // return array for system matrix elements
      int * const vxlId_devi, // return array for voxel ids
      int * const cnlId_devi, // return array for channel ids
      int const * const ml_devi,
      int const * const mlSize_devi,
      int * const truckDest_devi
     ) {
  // global id and global dim
  int const globalId  = threadIdx.x + blockIdx.x*blockDim.x;
  int const globalDim = blockDim.x*gridDim.x;
  
  __shared__ int nPassed_blck;
  __shared__ int truckCnlId_blck[TPB];
  __shared__ int truckVxlId_blck[TPB];
  __shared__ int truckDest_blck;
  
  // Master thread
  if(threadIdx.x == 0) {
    nPassed_blck = 0;
  }
  __syncthreads();
  
  GetId2vxlId<ConcreteVG>   f_vxlId(grid_const);
  GetId2listId<ConcreteVG>  f_listId(grid_const);
  IsInRange<ConcreteVG>     f_isInRange(grid_const, *mlSize_devi);
  GetIsLegal<ConcreteVG>    f_getIsLegal(grid_const, *mlSize_devi);
  for(int getId_thrd = globalId;
          f_isInRange(getId_thrd);
          getId_thrd += globalDim) {
    int vxlId_thrd  = f_vxlId( getId_thrd);
    int listId_thrd = f_listId(getId_thrd);
    int cnlId_thrd = -1;
    
    int writeOffset_thrd = -1;
    
    // Is getting another element legal?
    if(f_getIsLegal(vxlId_thrd, listId_thrd)) {

      // Get cnlId
      cnlId_thrd = ml_devi[listId_thrd];

      // Put this element to the test
      bool didPass_thrd = test<
                                T
                              , ConcreteVG
                              , ConcreteVGidx
                              , ConcreteVGidy
                              , ConcreteVGidz
                              , ConcreteMS
                              , ConcreteMSid0z
                              , ConcreteMSid0y
                              , ConcreteMSid1z
                              , ConcreteMSid1y
                              , ConcreteMSida
                              , ConcreteMSTrafo2CartCoordFirstPixel
                              , ConcreteMSTrafo2CartCoordSecndPixel>
                            (cnlId_thrd, vxlId_thrd);

      // Did it pass the test?
      if(didPass_thrd) {

        // Increase the count of passed elements in this block and get write
        // offset into shared mem
        writeOffset_thrd = atomicAdd(&nPassed_blck, 1);

        // Can this element be written to shared directly?
        if(writeOffset_thrd < TPB) {

          // Write element to shared
          truckCnlId_blck[writeOffset_thrd] = cnlId_thrd;
          truckVxlId_blck[writeOffset_thrd] = vxlId_thrd;
        }
      }
    }
    __syncthreads();

    // Is it time for a flush?
    if(nPassed_blck >= TPB) {
      
      // Master thread?
      if(threadIdx.x == 0) {
        truckDest_blck = atomicAdd(truckDest_devi, TPB);
        nPassed_blck -= TPB;
      }
      __syncthreads();
      
      // Calculate SM element and flush
      val_t sme_thrd = calcSme<
                              T
                            , ConcreteVG
                            , ConcreteVGidx
                            , ConcreteVGidy
                            , ConcreteVGidz
                            , ConcreteMS
                            , ConcreteMSid0z
                            , ConcreteMSid0y
                            , ConcreteMSid1z
                            , ConcreteMSid1y
                            , ConcreteMSida  
                            , ConcreteMSTrafo2CartCoordFirstPixel
                            , ConcreteMSTrafo2CartCoordSecndPixel >
                          ( truckCnlId_blck[threadIdx.x],
                            truckVxlId_blck[threadIdx.x]);
      
      sme_devi[  truckDest_blck+threadIdx.x]  = sme_thrd;
      cnlId_devi[truckDest_blck+threadIdx.x]  = truckCnlId_blck[threadIdx.x];
      vxlId_devi[truckDest_blck+threadIdx.x]  = truckVxlId_blck[threadIdx.x];
      __syncthreads();
      
      // Could this element NOT be written to shared before?
      if(writeOffset_thrd >= TPB) {
        
        writeOffset_thrd -= TPB;
        
        // Write element to shared
        truckCnlId_blck[writeOffset_thrd] = cnlId_thrd;
        truckVxlId_blck[writeOffset_thrd] = vxlId_thrd;
      }
    }
  }
  
  // Is a last flush necessarry?
  if(nPassed_blck > 0) {
    
    // Master thread?
    if(threadIdx.x == 0) {
      truckDest_blck = atomicAdd(truckDest_devi, nPassed_blck);
    }
    __syncthreads();
    
    // Does this thread take part?
    if(threadIdx.x < nPassed_blck) {
      
      // Calculate SM element and flush
      val_t sme_thrd = calcSme<
                              T
                            , ConcreteVG
                            , ConcreteVGidx
                            , ConcreteVGidy
                            , ConcreteVGidz
                            , ConcreteMS
                            , ConcreteMSid0z
                            , ConcreteMSid0y
                            , ConcreteMSid1z
                            , ConcreteMSid1y
                            , ConcreteMSida  
                            , ConcreteMSTrafo2CartCoordFirstPixel
                            , ConcreteMSTrafo2CartCoordSecndPixel >
                          (truckCnlId_blck[threadIdx.x],
                           truckVxlId_blck[threadIdx.x]);
      
      sme_devi[  truckDest_blck+threadIdx.x]  = sme_thrd;
      cnlId_devi[truckDest_blck+threadIdx.x]  = truckCnlId_blck[threadIdx.x];
      vxlId_devi[truckDest_blck+threadIdx.x]  = truckVxlId_blck[threadIdx.x];
      __syncthreads();
    }
  }
}

#endif /* GETSYSTEMMATRIXDEVICEONLY_CU */
