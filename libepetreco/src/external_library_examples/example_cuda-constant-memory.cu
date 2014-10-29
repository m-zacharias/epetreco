#include <stdio.h>
#include <iostream>
#include "CUDA_HandleError.hpp"

#define N 32

struct A {
  float a, b, c;
};



__constant__ float array_const[N];
__constant__ struct A a_const;

__global__ void kernel() {
  int globalId=threadIdx.x+blockIdx.x*blockDim.x;
  if(globalId<N)
    printf(
          "globalId: %i, array[globalId]: %f, a, b, c: %f, %f, %f\n",
          globalId, array_const[globalId], a_const.a, a_const.b, a_const.c);
}

int main() {
  
  float array[N];
  A a;
  a.a = 5.; a.b = 6.; a.c = 19.;
  for(int i=0; i<N; i++) {
    array[i]   = N-i;
  }
  std::cout << "sizeof(a)       on host:   " << sizeof(a) << std::endl
            << "sizeof(a_const) on device: " << sizeof(a_const) << std::endl;
    
  HANDLE_ERROR(cudaMemcpyToSymbol(array_const, array, sizeof(array_const)));
  HANDLE_ERROR(cudaMemcpyToSymbol(a_const, &a, sizeof(A)));
  kernel<<<1,N>>>();
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  return 0;
}