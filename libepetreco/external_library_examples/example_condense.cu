/* 
 * File:   condense.cu
 * Author: malte
 *
 * Created on 12. November 2014, 13:04
 */

#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>
#include <iomanip>

void HandleError( cudaError_t err, const char * file, int line )
{
  if(err != cudaSuccess)
    {
      std::cerr << file << "(" << line << "): error: " << cudaGetErrorString( err ) << std::endl;
      exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) { HandleError( err, __FILE__, __LINE__ ); }

#define SEED 1237
#define THRESHOLD 0.99
#define TPB 5
#define NBLOCKS 3
#define PTSIZE 2
#define TRUCKSIZE (PTSIZE*TPB)
//#define TRUCKSIZE 11
#define RINGSIZE (3*TRUCKSIZE)
#define SIZE NBLOCKS*TPB*PTSIZE


typedef float val_t;


__device__
void writeToRing( val_t const & elem, val_t * const ring, int const pos, int const ringSize ){
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

__global__
void condense( val_t * const globalElems, int * const globalStuff,
             int * globalTruckId ) {
  // Initialize
  int              globalId = threadIdx.x + blockDim.x*blockIdx.x;
  curandState      state;
  curand_init(SEED, globalId, 0, &state);
  val_t            threadRandoms[PTSIZE];
  __shared__ val_t blockElems[RINGSIZE];
  __shared__ int   blockWritePtr;
  __shared__ int   blockReadPtr;
  if(threadIdx.x ==0) {
    blockWritePtr = 0;
    blockReadPtr  = 0;
  }
  if(globalId==0)
    *globalTruckId = 0;
  
  // While there are elements to be tested, but at least once
  do {
    // While truck not yet full of survivors
    while((blockWritePtr/TRUCKSIZE)<1) {
      // Apply test to a fixed number of elements
      for(int i=0; i<PTSIZE; i++)
        threadRandoms[i] = curand_uniform(&state);
      val_t threadElems[PTSIZE];
      int _nThreadOut=0;
      for(int i=0; i<PTSIZE; i++) {
        if(threadRandoms[i]>=THRESHOLD) {
          threadElems[_nThreadOut]=threadRandoms[i];
          _nThreadOut++;
        }
      }

      // Publicate the number of elements that passed the test (survivors) within
      // the thread block
      __shared__ int nThreadOut[TPB];
      nThreadOut[threadIdx.x] = _nThreadOut;

      // Get position for writing the survivors
      __syncthreads();
      int _threadWriteStart = blockWritePtr;
      for(int i=0; i<threadIdx.x; i++)
        _threadWriteStart += nThreadOut[i];

      // Write survivors
      for(int i=0; i<_nThreadOut; i++)
        writeToRing(threadElems[i], blockElems, _threadWriteStart+i, RINGSIZE);

      // Increment blockWritePtr
      if(threadIdx.x == 0)
        for(int i=0; i<TPB; i++)
          incrementWritePtr(&blockWritePtr, nThreadOut[i], RINGSIZE);

      __syncthreads();
    }



    // --------------
    // CONTEXT SWITCH
    // --------------

    // Atomic register truck
    __shared__ int _globalTruckId;
    if(threadIdx.x==0)
      _globalTruckId = atomicAdd(globalTruckId, 1);
    __syncthreads();
    
    // Calculate stuff
    __shared__ val_t blockStuff[TRUCKSIZE];
    int truckElemId = threadIdx.x;
    while(truckElemId < TRUCKSIZE) {
      val_t truckElem(0.);
      readFromRing(&truckElem, blockElems, blockReadPtr+truckElemId, RINGSIZE);
      printf("Calculate stuff: block %i, truckElem %i: %f\n", blockIdx.x,
            truckElemId, truckElem);
      blockStuff[truckElemId] = floor(10. * truckElem);
      truckElemId += blockDim.x;
    }

    // Write stuff to global
    truckElemId = threadIdx.x;
    while(truckElemId < TRUCKSIZE) {
      val_t truckElem(0.);
      readFromRing(&truckElem, blockElems, blockReadPtr+truckElemId, RINGSIZE);
      printf("Write stuff: block %i: Write (%f, %f) to %i\n", blockIdx.x,
            truckElem, blockStuff[truckElemId],
            (truckElemId + _globalTruckId*TRUCKSIZE));
      globalElems[truckElemId + _globalTruckId*TRUCKSIZE] = truckElem;
      globalStuff[truckElemId + _globalTruckId*TRUCKSIZE] = blockStuff[truckElemId];
      truckElemId += blockDim.x;
    }

    // Increment blockReadPtr
    incrementReadPtr(&blockReadPtr, TRUCKSIZE, RINGSIZE);
  } while(false);
}



int main(int argc, char** argv) {
  val_t elems_host[SIZE];
  int   stuff_host[SIZE];
  for(int i=0; i<SIZE; i++) {
    elems_host[i] = 0.;
    stuff_host[i] = 0;
  }
  
  val_t * elems_devi   = NULL;
  int   * stuff_devi   = NULL;
  int   * truckId_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&elems_devi, sizeof(*elems_devi) * SIZE));
  HANDLE_ERROR(cudaMalloc((void**)&stuff_devi, sizeof(*stuff_devi) * SIZE));
  HANDLE_ERROR(cudaMalloc((void**)&truckId_devi, sizeof(*truckId_devi)));
  HANDLE_ERROR(cudaMemcpy(elems_devi, elems_host, sizeof(*elems_devi) * SIZE, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(stuff_devi, stuff_host, sizeof(*stuff_devi) * SIZE, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  condense<<<NBLOCKS, TPB>>>(elems_devi, stuff_devi, truckId_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  HANDLE_ERROR(cudaMemcpy(elems_host, elems_devi, sizeof(*elems_host) * SIZE, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(stuff_host, stuff_devi, sizeof(*stuff_host) * SIZE, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  for(int i=0; i<SIZE; i++) {
    std::cout << "elem: " << std::setw(8) << elems_host[i] << ", stuff: " << stuff_host[i]
              << std::endl;
  }
  
  cudaFree(elems_devi);
  cudaFree(stuff_devi);
  
  exit(EXIT_SUCCESS);
}

