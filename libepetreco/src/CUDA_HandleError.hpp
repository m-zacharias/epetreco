#ifndef CUDA_HANDLE_ERROR
#define CUDA_HANDLE_ERROR

//#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

//void HandleError( cudaError_t err, const char * file, int line )
//{
//  if(err != cudaSuccess)
//    {
//    printf("%s(%d): error: %s\n", file, line, cudaGetErrorString( err ) );
//    exit( EXIT_FAILURE );
//    }
//}
void HandleError( cudaError_t err, const char * file, int line )
{
  if(err != cudaSuccess)
    {
      std::cerr << file << "(" << line << "): error: " << cudaGetErrorString( err ) << std::endl;
      exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) { HandleError( err, __FILE__, __LINE__ ); }
//#define HANDLE_ERROR( err ) { err; }

#endif  // #ifndef CUDA_HANDLE_ERROR
