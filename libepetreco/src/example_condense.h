/* 
 * File:   example_condense.h
 * Author: malte
 *
 * Created on 26. November 2014, 15:48
 */

#ifndef EXAMPLE_CONDENSE_H
#define	EXAMPLE_CONDENSE_H



#include "example_condense_defines.h"
#include <cuda.h>
#include <curand_kernel.h>
//#include <stdlib.h>
//#include <stdio.h>
  
typedef float val_t;

__device__
void writeToRing(
      val_t const & elem, val_t * const ring, int const pos,
      int const ringSize ) {
  ring[pos%ringSize] = elem;
}

__device__
void readFromRing(
      val_t * const target, val_t const * const ring, int const pos,
      int const ringSize ) {
  *target = ring[pos%ringSize];
}

__device__
void incrementReadPtr(
      int * const readPtr, int const inc, int const ringSize ) {
  *readPtr = (*readPtr+inc)%ringSize;
}

__device__
void incrementWritePtr(
      int * const writePtr, int const inc, int const ringSize ) {
  *writePtr = (*writePtr+inc)%ringSize;
}

__device__
bool test( val_t const & elem ) {
//  return (elem>THRESHOLD);
  return true;
}

template<typename State>
__device__
void init_elems( State * const, int const );

template<>
__device__
void init_elems<curandState>( curandState * const state_thrd, int const start ){
  curand_init(SEED, 0, start, state_thrd);
}

struct OwnState {
  int i;
};

template<>
__device__
void init_elems<OwnState>( OwnState * const state_thrd, int const start ){
  state_thrd->i = start;
}

template<typename State>
__device__
val_t get_elem( State * const );

template<>
__device__
val_t get_elem<curandState>( curandState * const state_thrd) {
  return curand_uniform(state_thrd);
}

template<>
__device__
val_t get_elem<OwnState>( OwnState * const state_thrd) {
  state_thrd->i += 1;
  return (state_thrd->i)-1;
}

__global__
void condense(
      val_t * const passed_devi, int * truckDest_devi ) {
  
  // Initialize
  int              globalId = threadIdx.x + blockDim.x*blockIdx.x;
  val_t            boat_thrd[BOATSIZE];
  __shared__ val_t ring_blck[RINGSIZE];
  __shared__ int   writePos_blck;
  __shared__ int   readPos_blck;
  __shared__ int   nLanded_blck;
  
  if(threadIdx.x == 0) {
    writePos_blck = 0;
    readPos_blck  = 0;
    nLanded_blck  = 0;
  }
  
//  curandState      state_thrd;
  OwnState         state_thrd;
  int              nWaiting_thrd(0);
  if(globalId<LATER_ID) {
    init_elems(&state_thrd, globalId * FIRST_LEN);
    nWaiting_thrd = FIRST_LEN;
  } else {
    init_elems(&state_thrd, (LATER_ID * FIRST_LEN) + ((globalId-LATER_ID) * LATER_LEN));
    nWaiting_thrd = LATER_LEN;
  }
    
  __syncthreads();
  
  // While there are elements underway...
  while((nWaiting_thrd + nLanded_blck)>0) {
    
    // ...put those that survive in another truck.
    // While truck should wait for more survivors...
    while((nLanded_blck<TRUCKSIZE)&&(nWaiting_thrd>0)) {
      
      // ...load another boat. Count, how many elements enter the boat.
      int nInBoat_thrd = 0;
      // While boat should wait for more elements...
      while((nInBoat_thrd<BOATSIZE)&&(nWaiting_thrd>0)) {
        // ...put another element in the boat. Keep counting and tick off the
        // latest element that entered the boat.
        boat_thrd[nInBoat_thrd] = get_elem(&state_thrd);
        nInBoat_thrd++;
        nWaiting_thrd--;
      }
      
      // Send the latest boat through the test. Count, how many elements pass
      // the test.
      val_t passed_thrd[BOATSIZE];
      int nPassed_thrd = 0;
      // For all elements that were put in the boat...
      for(int i=0; i<nInBoat_thrd; i++) {
        // ...put the current element to the test. Keep counting.
        if(test(boat_thrd[i])) {
          passed_thrd[nPassed_thrd]=boat_thrd[i];
          nPassed_thrd++;
        }
      }

      // Publicate the number of survivors on the latest boat within
      // the thread block.
      // I.e. there's a man with the truck, who tells the boats where in the
      // truck they should sit their survivors. Tell this man the number of
      // survivors on the latest boat - he depends upon that information to do
      // his job properly.
      __shared__ int nPassed_blck[TPB];
      nPassed_blck[threadIdx.x] = nPassed_thrd;

      // Ask the man, where in the truck to sit the survivors from the latest
      // boat.
      __syncthreads();
      int writeOffset_thrd = 0;
      for(int i=0; i<threadIdx.x; i++)
        writeOffset_thrd += nPassed_blck[i];

      // Put the survivors in the truck.
      for(int i=0; i<nPassed_thrd; i++) {
        writeToRing(passed_thrd[i], ring_blck, writePos_blck+writeOffset_thrd+i, RINGSIZE);
//        printf("thread %i in block %i puts %f in the truck\n", threadIdx.x, blockIdx.x, passed_thrd[i]);
        // CONFIRMED: Each survivor is put into exactly one truck
//        val_t surv(0.);
//        readFromRing(&surv, ring_blck, writePos_blck+writeOffset_thrd+i, RINGSIZE);
//        printf("survivor %f\n", surv);
        // CONFIRMED: Reading directly after writing gives back all survivors
      }
      __syncthreads();
      
      // Increment writePos_blck, update number of survivors in ring
      if(threadIdx.x == 0) {
        for(int i=0; i<blockDim.x; i++) {
          incrementWritePtr(&writePos_blck, nPassed_blck[i], RINGSIZE);
          nLanded_blck += nPassed_blck[i];
        }
      }
      __syncthreads();
      
    } // Truck finished waiting for survivors
    
    // --------------
    // CONTEXT SWITCH
    // --------------
    __syncthreads();
    __shared__ int nInTruck_blck;
    if(threadIdx.x == 0) {
      if(nLanded_blck>=TRUCKSIZE) {
        nInTruck_blck = TRUCKSIZE;
      } else {
        nInTruck_blck = nLanded_blck;
      }
    }
    __syncthreads();

    // Atomic register truck
    __shared__ int truckDest_blck;
    if(threadIdx.x==0) {
      truckDest_blck = atomicAdd(truckDest_devi, nInTruck_blck);
    }
    __syncthreads();

    // Write to global
    int truckElemId = threadIdx.x;
    while(truckElemId < nInTruck_blck) {
      val_t truckElem(0.);
      readFromRing(&truckElem, ring_blck, readPos_blck+truckElemId, RINGSIZE);
//      printf("Write stuff: block %i: Write (%f, %f) to %i\n", blockIdx.x,
//            truckElem, blockStuff[truckElemId],
//            (truckElemId + truckDest_blck));
      // This causes __global__ write error sometimes
      passed_devi[truckElemId + truckDest_blck] = truckElem;
      truckElemId += blockDim.x;
    }

    __syncthreads();
    if(threadIdx.x == 0) {
      // Increment readPos_blck
      incrementReadPtr(&readPos_blck, nInTruck_blck, RINGSIZE);

      // Update number of elements in ring
      nLanded_blck -= nInTruck_blck;
    }
    __syncthreads();
  } // there are no more elements underway
}



#endif	/* EXAMPLE_CONDENSE_H */

