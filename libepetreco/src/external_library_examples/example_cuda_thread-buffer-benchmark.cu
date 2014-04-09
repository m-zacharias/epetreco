/* This is a test. Threads produce numbers in a loop which are then written into
 * global memory. Three different methods are implemented:
 *  - thread produces each number locally and writes it immediately
 *  - thread produces numbers in a single buffer, buffer is then written into
 *    global memory
 *  - thread has two buffers, one buffer is used for number production while the
 *    other is written into global memory
 */
#include <cuda.h>
#include <curand_kernel.h>
#include "CUDA_HandleError.hpp"
#include "FileTalk.hpp"

#define BSIZE 64

__global__
void db_kernel( float * const a, int const size )
{
  // Initialize curandState
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  curand_init(1234, id, 0, &state);

  // Initialize two buffers
  float localmem0[BSIZE];
  float localmem1[BSIZE];
  
  int memid = 0;
  float * localmem = localmem0;
  
  // Iterate
  while(memid < size)
  {
    // set value in current buffer
    localmem[memid%BSIZE] = curand_uniform(&state) * curand_uniform(&state) / curand_uniform(&state);
    
    // if at end of buffer / end of global array
    if(memid%BSIZE==BSIZE-1 || memid==size-1)
    {
      int write_size = memid%BSIZE+1;
      int start      = memid-write_size+1;
      int end        = memid+1;
      
      // if current buffer is 1st buffer
      if(localmem==localmem0)
      {
        // set 2nd buffer as current buffer
        localmem = localmem1;
        // copy elements from 1st buffer to global
        for(int i=start; i<end; i++)
        {
          a[i] = localmem0[i%BSIZE];
                 localmem0[i%BSIZE] = 0.;
        }
      }
      // if current buffer is 2nd buffer
      else if(localmem==localmem1)
      {
        // set 1st buffer as current buffer
        localmem = localmem0;
        // copy elements from 2nd buffer to global
        for(int i=start; i<end; i++)
        {
          a[i] = localmem1[i%BSIZE];
                 localmem1[i%BSIZE] = 0.;
        }
      }
    }
    memid++;
  }
}

__global__
void b_kernel( float * const a, int const size )
{
  // Initialize curandState
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  curand_init(1234, id, 0, &state);
  
  // Initialize buffer
  float localmem[BSIZE];

  int memid=0;

  while(memid < size)
  {
    // set value in current buffer
    localmem[memid%BSIZE] = curand_uniform(&state) * curand_uniform(&state) / curand_uniform(&state);
    
    // if at end of buffer / end of global array
    if(memid%BSIZE==BSIZE-1 || memid==size-1)
    {
      int write_size = memid%BSIZE+1;
      int start      = memid-write_size+1;
      int end        = memid+1;
      
      // copy elements from 1st buffer to global
      for(int i=start; i<end; i++)
      {
        a[i] = localmem[i%BSIZE];
               localmem[i%BSIZE] = 0.;
      }
    }
    memid++;
  }
}

__global__
void n_kernel( float * const a, int const size )
{
  // Initialize curandState
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  curandState state;
  curand_init(1234, id, 0, &state);

  int memid = 0;

  while(memid < size)
  {
    float temp = curand_uniform(&state) * curand_uniform(&state) / curand_uniform(&state);
    a[memid] = temp;
    memid++;
  }
}



#define NBLOCKS 1
#define NTHREADS 1024
#define SIZE 65536

#include <iostream>
int main()
{
  SAYLINE( __LINE__+1 );
  cudaSetDevice(0);

  // Create timers
  cudaEvent_t start_db, stop_db;
  cudaEventCreate(&start_db); cudaEventCreate(&stop_db);
  
  cudaEvent_t start_b,  stop_b;
  cudaEventCreate(&start_b); cudaEventCreate(&stop_b);
  
  cudaEvent_t start_n,  stop_n;
  cudaEventCreate(&start_n); cudaEventCreate(&stop_n);
  
  // Allocate global memory
  float * a;
  SAYLINE( __LINE__+1 );
  HANDLE_ERROR( cudaMalloc((void**)&a, SIZE*sizeof(float)) );

  // Time measurement
  SAYLINE( __LINE__-1 );
  cudaEventRecord(start_db, 0);
  db_kernel<<<NBLOCKS, NTHREADS>>>( a, SIZE );
  cudaEventRecord( stop_db, 0);
  cudaThreadSynchronize();
  float db_time;
  cudaEventElapsedTime(&db_time, start_db, stop_db);
  
  cudaEventRecord( start_b, 0);
  b_kernel<<<NBLOCKS, NTHREADS>>> ( a, SIZE );
  cudaEventRecord(  stop_b, 0);
  cudaThreadSynchronize();
  float  b_time;
  cudaEventElapsedTime( &b_time,  start_b,  stop_b);
  
  cudaEventRecord( start_n, 0);
  n_kernel<<<NBLOCKS, NTHREADS>>> ( a, SIZE );
  cudaEventRecord(  stop_n, 0);
  cudaThreadSynchronize();
  float n_time;
  cudaEventElapsedTime( &n_time,  start_n,  stop_n);

  // Print Result
  std::cout << "db: " << db_time << " ms" << std::endl;
  std::cout << " b: " <<  b_time << " ms" << std::endl;
  std::cout << " n: " <<  n_time << " ms" << std::endl;
  
  return 0;
}
