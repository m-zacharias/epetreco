#include <cuda_runtime.h>
#include <cublasXt.h>

#include <iostream>
#include <cassert>

#include "FileTalk.hpp"

typedef float val_t;

#define N 40000

#define CHECK( b, m ) { if((b)){std::cerr<<(m)<<std::endl;exit(EXIT_FAILURE);} }

int main()
{
  cudaError_t       cudaStat;
  cublasStatus_t    cublasStat;
  cublasXtHandle_t  cublasHandle;
 
  SAYLINE(__LINE__+1);
  /* Initialize cuBLAS-XT */
  cublasStat = cublasXtCreate(&cublasHandle);
  CHECK(cublasStat!=CUBLAS_STATUS_SUCCESS, "cuBLAS-XT initialization failed");
  
  SAYLINE(__LINE__+1);
  /* Select devices */
  int devices[2] = {0,1};
  cublasStat = cublasXtDeviceSelect(cublasHandle, 2, devices);
  CHECK(cublasStat!=CUBLAS_STATUS_SUCCESS, "cuBLAS-XT device selection failed");
  
  SAYLINE(__LINE__+1);
  /* Allocate memory */
  val_t * A, * B, *C ;
  size_t SIZE(N*N);
  
  SAYLINE(__LINE__+1);
  cudaStat = cudaMallocManaged(&A, SIZE);
  SAYLINE(__LINE__+1);
  CHECK(cudaStat!=cudaSuccess, "Memory allocation failed");
  SAYLINES(__LINE__+1,__LINE__+2);
  for(size_t linId=0; linId<SIZE; linId++)
  {
    if(linId%10000000==0||(linId>400000000)) std::cout << linId << std::endl;
    A[linId] = linId;
  }

  SAYLINE(__LINE__+1);
  cudaStat = cudaMallocManaged(&B, SIZE);
  SAYLINE(__LINE__+1);
  CHECK(cudaStat!=cudaSuccess, "Memory allocation failed");
  SAYLINES(__LINE__+1,__LINE__+3);
  for(size_t linId=0; linId<SIZE; linId++)
    if((linId/N)==(linId%N))
      B[linId] = 1.;

  SAYLINE(__LINE__+1);
  cudaStat = cudaMallocManaged(&C, SIZE);
  SAYLINE(__LINE__+1);
  CHECK(cudaStat!=cudaSuccess, "Memory allocation failed");
  
  SAYLINE(__LINE__+1);
  /* Matrix Matrix Multiplication */
  val_t alpha = 1., beta = 0.;
  cublasStat = cublasXtSgemm( cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                N, N, N,
                                &alpha,
                                A, N,
                                B, N,
                                &beta,
                                C, N);

  SAYLINE(__LINE__+1);
  /* Assert */
  for(size_t linId=0; linId<SIZE; linId++)
    assert(C[linId]==A[linId]);

  SAYLINE(__LINE__+1);
  /* Free resources */
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

  return 0;
}
