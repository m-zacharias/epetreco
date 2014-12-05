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
void writeToRing( val_t const & elem, val_t * const ring, int const pos, int const ringSize ) {
  ring[pos%ringSize] = elem;
}

__device__
void readFromRing( val_t * const target, val_t const * const ring, int const pos, int const ringSize ) {
  *target = ring[pos%ringSize];
}

__device__
void incrementReadPtr( int * const readPtr, int const inc, int const ringSize ) {
  *readPtr = (*readPtr+inc)%ringSize;
}

__device__
void incrementWritePtr( int * const writePtr, int const inc, int const ringSize ) {
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
void init_elems<curandState>( curandState * const state, int const start ){
  curand_init(SEED, 0, start, state);
}

struct OwnState {
  int i;
};

template<>
__device__
void init_elems<OwnState>( OwnState * const state, int const start ){
  state->i = start;
}

template<typename State>
__device__
val_t get_elem( State * const );

template<>
__device__
val_t get_elem<curandState>( curandState * const state) {
  return curand_uniform(state);
}

template<>
__device__
val_t get_elem<OwnState>( OwnState * const state) {
  ++(state->i);
  return (state->i)-1;
//  return threadIdx.x + blockDim.x*blockIdx.x;
}

__global__
void condense(
      val_t * const globalElems, int * const globalStuff,
      int * const globalBlock, int * globalWriteId ) {
  
  // Initialize
  int              globalId = threadIdx.x + blockDim.x*blockIdx.x;
  val_t            threadRandoms[T_TEST_SIZE];
  __shared__ val_t blockElems[RINGSIZE];
  __shared__ int   blockWritePtr;
  __shared__ int   blockReadPtr;
  __shared__ int   nInRing;
  
  if(threadIdx.x ==0) {
    blockWritePtr = 0;
    blockReadPtr  = 0;
    nInRing       = 0;
  }
  
//  curandState      state;
  OwnState         state;
  int              nWaiting(0);
  if(globalId<LATER_ID) {
    nWaiting = FIRST_LEN;
    init_elems(&state, globalId * FIRST_LEN);
  } else {
    nWaiting = LATER_LEN;
    init_elems(&state, (LATER_ID * FIRST_LEN) + ((globalId-LATER_ID) * LATER_LEN));
  }
    
  __syncthreads();
  
  // While there are elements underway...
  while((nWaiting + nInRing)>0) {
    
    // ...put those that survive in another truck.
    // While truck should wait for more survivors...
    while((nInRing<TRUCKSIZE)&&(nWaiting>0)) {
      
      // ...load another boat. Count, how many elements enter the boat.
      int nGotForTest = 0;
      // While boat should wait for more elements...
      while((nGotForTest<T_TEST_SIZE)&&(nWaiting>0)) {
        // ...put another element in the boat. Keep counting and tick off the
        // latest element that entered the boat.
        threadRandoms[nGotForTest] = get_elem(&state);
//        printf("Got: thread %i, block %i: %f\n", threadIdx.x, blockIdx.x, threadRandoms[nGotForTest]);
        nGotForTest++;
        nWaiting--;
      }
      
      // Send the latest boat through the test. Count, how many elements pass
      // the test.
      val_t threadPassed[T_TEST_SIZE];
      int _nThreadPassed=0;
      // For all elements that were put in the boat...
      for(int i=0; i<nGotForTest; i++) {
        // ...put the current element to the test. Keep counting.
        if(test(threadRandoms[i])) {
          threadPassed[_nThreadPassed]=threadRandoms[i];
          printf("Passed: thread %i, block %i: %f\n", threadIdx.x, blockIdx.x, threadPassed[_nThreadPassed]);
          _nThreadPassed++;
        }
      }

      // Publicate the number of survivors on the latest boat within
      // the thread block.
      // I.e. there's a man with the truck, who tells the boats where in the
      // truck they should sit their survivors. Tell this man the number of
      // survivors on the latest boat - he depends upon that information to do
      // his job properly.
      __shared__ int nThreadPassed[TPB];
      nThreadPassed[threadIdx.x] = _nThreadPassed;

      // Ask the man, where in the truck to sit the survivors from the latest
      // boat.
      __syncthreads();
      int _threadWriteStart = 0;
      for(int i=0; i<threadIdx.x; i++)
        _threadWriteStart += nThreadPassed[i];

      // Put the survivors in the truck.
      for(int i=0; i<_nThreadPassed; i++) {
        writeToRing(threadPassed[i], blockElems, blockWritePtr+_threadWriteStart+i, RINGSIZE);
//        printf("thread %i in block %i puts %f in the truck\n", threadIdx.x, blockIdx.x, threadPassed[i]);
      }
      __syncthreads();
      
      // Increment blockWritePtr, update number of survivors in ring
      if(threadIdx.x == 0) {
        for(int i=0; i<blockDim.x; i++) {
          incrementWritePtr(&blockWritePtr, nThreadPassed[i], RINGSIZE);
          nInRing += nThreadPassed[i];
        }
      }
      __syncthreads();
      
    } // Truck finished waiting for survivors

    
    // --------------
    // CONTEXT SWITCH
    // --------------

    __shared__ int nBlockOut;
    if(threadIdx.x == 0) {
      if(nInRing>=TRUCKSIZE) {
        nBlockOut = TRUCKSIZE;
      } else {
        nBlockOut = nInRing;
      }
    }
    __syncthreads();
    if(threadIdx.x==0) {
      printf("Latest truck (%i) contains: ", nBlockOut);
    }
    for(int i=0; i<nBlockOut; i++) {
      val_t truckElem(0);
      readFromRing(&truckElem, blockElems, blockReadPtr+i, RINGSIZE);
      printf("%f, ", truckElem);
    }
    printf("\n");
    
    
    // Atomic register truck
    __shared__ int _globalWriteId;
    if(threadIdx.x==0)
      _globalWriteId = atomicAdd(globalWriteId, nBlockOut);
    __syncthreads();
    
    // Calculate stuff
    __shared__ val_t blockStuff[TRUCKSIZE];
    int truckElemId = threadIdx.x;
    while(truckElemId < nBlockOut) {
      val_t truckElem(0.);
      readFromRing(&truckElem, blockElems, blockReadPtr+truckElemId, RINGSIZE);
//      printf("Calculate stuff: block %i, truckElem %i: %f\n", blockIdx.x,
//            truckElemId, truckElem);
      blockStuff[truckElemId] = floor(10. * truckElem);
      truckElemId += blockDim.x;
    }

    // Write stuff to global
    truckElemId = threadIdx.x;
    while(truckElemId < nBlockOut) {
      val_t truckElem(0.);
      readFromRing(&truckElem, blockElems, blockReadPtr+truckElemId, RINGSIZE);
//      printf("Write stuff: block %i: Write (%f, %f) to %i\n", blockIdx.x,
//            truckElem, blockStuff[truckElemId],
//            (truckElemId + _globalWriteId));
      globalElems[truckElemId + _globalWriteId] = truckElem;
      globalStuff[truckElemId + _globalWriteId] = blockStuff[truckElemId];
      globalBlock[truckElemId + _globalWriteId] = blockIdx.x;
      truckElemId += blockDim.x;
    }

    // Increment blockReadPtr
    incrementReadPtr(&blockReadPtr, nBlockOut, RINGSIZE);
    
    // Update number of elements in ring
    nInRing -= nBlockOut;
  }
}



#endif	/* EXAMPLE_CONDENSE_H */

