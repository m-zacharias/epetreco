#ifndef CUDA_HANDLE_ERROR
#define CUDA_HANDLE_ERROR

#include <cstdio>

void HandleError( cudaError_t err, const char * file, int line )
{
  if(err != cudaSuccess)
    {
    printf("%s(%d): error: %s\n", file, line, cudaGetErrorString( err ) );
    exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) {HandleError( err, __FILE__, __LINE__ );};

#endif  // #ifndef CUDA_HANDLE_ERROR
