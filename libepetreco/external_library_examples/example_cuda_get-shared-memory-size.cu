/* Find out cuda shared memory
 */
#include <cuda_runtime.h>
#include "CUDA_HandleError.hpp"
#include <iostream>

int main()
{
  cudaDeviceProp prop;
  HANDLE_ERROR( cudaGetDeviceProperties( &prop, 0 ) );
  size_t mem = prop.sharedMemPerBlock;
  std::cout << "Maximum shared memory: "
            << mem / 1024 << "kB " << mem % 1024 << "B" << std::endl;

  return 0;
}
