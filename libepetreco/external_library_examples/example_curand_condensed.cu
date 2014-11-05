
// 2014-03-28
// http://docs.nvidia.com/cuda/curand/device-api-overview.html#device-api-example
/*
 * This program uses the device CURAND API to calculate what 
 * proportion of pseudo-random ints have low bit set.
 * It then generates uniform results to calculate how many
 * are greater than .5.
 * It then generates  normal results to calculate how many 
 * are within one standard deviation of the mean.
 */
#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand_kernel.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); \
    return EXIT_FAILURE;}} while(0)

__global__ void setup_kernel( curandState * state )
{
  int id = threadIdx.x + blockIdx.x * 64;
  /* Each thread gets same seed, a different sequence 
     number, no offset */
  curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(
      curandState *state,
      int n, 
      unsigned int *result)
{
  int id = threadIdx.x + blockIdx.x * 64;
  int count = 0;
  unsigned int x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[id];
  /* Generate pseudo-random unsigned ints */
  for(int i = 0; i < n; i++) {
    x = curand(&localState);
    /* Check if low bit set */
    if(x & 1) {
      count++;
    }
  }
  /* Copy state back to global memory */
  state[id] = localState;
  /* Store results */
  result[id] += count;
}

__global__ void generate_uniform_kernel(
      curandState *state,
      int n, 
      unsigned int *result)
{
  int id = threadIdx.x + blockIdx.x * 64;
  unsigned int count = 0;
  float x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[id];
  /* Generate pseudo-random uniforms */
  for(int i = 0; i < n; i++) {
    x = curand_uniform(&localState);
    /* Check if > .5 */
    if(x > .5) {
      count++;
    }
  }
  /* Copy state back to global memory */
  state[id] = localState;
  /* Store results */
  result[id] += count;
}

__global__ void generate_normal_kernel(
      curandState *state,
      int n, 
      unsigned int *result)
{
  int id = threadIdx.x + blockIdx.x * 64;
  unsigned int count = 0;
  float2 x;
  /* Copy state to local memory for efficiency */
  curandState localState = state[id];
  /* Generate pseudo-random normals */
  for(int i = 0; i < n/2; i++) {
    x = curand_normal2(&localState);
    /* Check if within one standard deviaton */
    if((x.x > -1.0) && (x.x < 1.0)) {
      count++;
    }
    if((x.y > -1.0) && (x.y < 1.0)) {
      count++;
    }
  }
  /* Copy state back to global memory */
  state[id] = localState;
  /* Store results */
  result[id] += count;
}



int main(int argc, char *argv[])
{
  int i;
  unsigned int total;
  curandState *devStates;
  unsigned int *devResults, *hostResults;
  int sampleCount = 10000;
  bool doubleSupported = 0;
  int device;
  struct cudaDeviceProp properties;  

  /* check for double precision support */
  CUDA_CALL(cudaGetDevice(&device));
  CUDA_CALL(cudaGetDeviceProperties(&properties,device));
  if ( properties.major >= 2 || (properties.major == 1 && properties.minor >= 3) ) {
    doubleSupported = 1;
  }

  /* Allocate space for results on host */
  hostResults = (unsigned int *)calloc(64 * 64, sizeof(int));

  /* Allocate space for results on device */
  CUDA_CALL(cudaMalloc((void **)&devResults, 64 * 64 * 
        sizeof(unsigned int)));

  /* Set results to 0 */
  CUDA_CALL(cudaMemset(devResults, 0, 64 * 64 * 
        sizeof(unsigned int)));

  /* Allocate space for prng states on device */
  CUDA_CALL(cudaMalloc((void **)&devStates, 64 * 64 * 
        sizeof(curandState)));
  
  /* Setup prng states */
  setup_kernel<<<64, 64>>>(devStates);
  
  /* Generate and use pseudo-random  */
  generate_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
  
  
  
  /* Copy device memory to host */
  CUDA_CALL(cudaMemcpy(hostResults, devResults, 64 * 64 * 
    sizeof(unsigned int), cudaMemcpyDeviceToHost));

  /* Show result */
  total = 0;
  for(i = 0; i < 64 * 64; i++) {
    total += hostResults[i];
  }
  printf("Fraction with low bit set was %10.13f\n", 
    (float)total / (64.0f * 64.0f * sampleCount * 50.0f));
    
  /* Set results to 0 */
  CUDA_CALL(cudaMemset(devResults, 0, 64 * 64 * 
        sizeof(unsigned int)));

  /* Generate and use uniform pseudo-random  */
  generate_uniform_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
  
  

  /* Copy device memory to host */
  CUDA_CALL(cudaMemcpy(hostResults, devResults, 64 * 64 * 
    sizeof(unsigned int), cudaMemcpyDeviceToHost));

  /* Show result */
  total = 0;
  for(i = 0; i < 64 * 64; i++) {
    total += hostResults[i];
  }
  printf("Fraction of uniforms > 0.5 was %10.13f\n", 
    (float)total / (64.0f * 64.0f * sampleCount * 50.0f));
  /* Set results to 0 */
  CUDA_CALL(cudaMemset(devResults, 0, 64 * 64 * 
        sizeof(unsigned int)));

  /* Generate and use uniform pseudo-random  */
  generate_normal_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
  
  

  /* Copy device memory to host */
  CUDA_CALL(cudaMemcpy(hostResults, devResults, 64 * 64 * 
    sizeof(unsigned int), cudaMemcpyDeviceToHost));

  /* Show result */
  total = 0;
  for(i = 0; i < 64 * 64; i++) {
    total += hostResults[i];
  }
  printf("Fraction of normals within 1 standard deviation was %10.13f\n", 
    (float)total / (64.0f * 64.0f * sampleCount * 50.0f));

  /* Cleanup */
  CUDA_CALL(cudaFree(devStates));
   
  CUDA_CALL(cudaFree(devResults));
  free(hostResults);
  printf("^^^^ kernel_example PASSED\n");
  return EXIT_SUCCESS;
}
