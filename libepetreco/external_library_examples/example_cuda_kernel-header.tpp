#include "example_cuda_kernel-header.hpp"

template< typename T >
__global__
void add(
      T * const a_devi, T * const b_devi, T * const result_devi,
      int const n_devi ) {
  int const gid   = threadIdx.x + blockDim.x*blockIdx.x;
  int const gdim  = blockDim.x * gridDim.x;
  
  for(int id=gid; id<(n+TPB-1)/TPB; gid+=gdim) {
    if(id<n_devi) {
      result_devi[id] = a_devi[id] + b_devi[id];
    }
  }
}
