#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>

#define M 6
#define N 4

#define IDX2C(i,j,ld) (((j)*(ld))+i)

int main()
{
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;

  float * A_host = 0;
  float * A_devi;
  float * x_host = 0;
  float * x_devi;
  float * y_host = 0;
  float * y_devi;
  
  /* Host memory alocation */
  A_host = (float *) malloc(M*N*sizeof(*A_host));
  if(!A_host) {
    std::cerr << "Host memory allocation failed" << std::endl;
    return EXIT_FAILURE;
  }
  for(int j=0; j<N; j++) {
    for(int i=0; i<M; i++) {
      A_host[IDX2C(i,j,M)] = (float)(i * M + j + 1);
    }
  }
  x_host = (float *) malloc(N*sizeof(*A_host));
  if(!A_host) {
    std::cerr << "Host memory allocation failed" << std::endl;
    return EXIT_FAILURE;
  }
  for(int j=0; j<N; j++) {
    x_host[j] = j;
  }
  y_host = (float *) malloc(M*sizeof(*A_host));
  if(!A_host) {
    std::cerr << "Host memory allocation failed" << std::endl;
    return EXIT_FAILURE;
  }
  for(int i=0; i<M; i++) {
    y_host[i] = 0;
  }
  
  
  
  /* Device memory allocation */
  cudaStat = cudaMalloc((void**)&A_devi, M*N*sizeof(*A_host));
  if(cudaStat != cudaSuccess) {
    std::err << "Device memory allocation failed" << std::endl;
    return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc((void**)&x_devi, N*sizeof(*A_host));
  if(cudaStat != cudaSuccess) {
    std::err << "Device memory allocation failed" << std::endl;
    return EXIT_FAILURE;
  }
  cudaStat = cudaMalloc((void**)&y_devi, M*sizeof(*A_host));
  if(cudaStat != cudaSuccess) {
    std::err << "Device memory allocation failed" << std::endl;
    return EXIT_FAILURE;
  }
  
  
  
  /* CUBLAS initialization */
  stat = cublasCreate(&handle);
  if(stat != CUBLASS_STATUS_SUCCESS) {
    std::err << "CUBLAS initialization failed" << std::endl;
    return EXIT_FAILURE;
  }
  
  
  
  /* Data download */
  stat = cublasSetMatrix(M, N, sizeof(*A_host), A_host, M, A_devi, M);
  if(stat != CUBLASS_STATUS_SUCCESS) {
    std::err << "Data download failed" << std::endl;
    cudaFree A_devi;
    cudaFree x_devi;
    cudaFree y_devi;
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  stat = cublasSetVector(N, sizeof(*A_host), x_host, 1, x_devi, 1);
  if(stat != CUBLASS_STATUS_SUCCESS) {
    std::err << "Data download failed" << std::endl;
    cudaFree A_devi;
    cudaFree x_devi;
    cudaFree y_devi;
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
//  stat = cublasSetVector(M, sizeof(*A_host), y_host, 1, y_devi, 1);
//  if(stat != CUBLASS_STATUS_SUCCESS) {
//    std::err << "Data download failed" << std::endl;
//    cudaFree A_devi;
//    cudaFree x_devi;
//    cudaFree y_devi;
//    cublasDestroy(handle);
//    return EXIT_FAILURE;
//  }
  
  
  
  /* Matrix Vector Multiplication */
  stat = cublasSgemv(M, N, *alpha, A_devi, M, x_devi, 1, *beta, y_devi, 1);
  
  
  
  /* Data upload */
  stat = cublasGetVector(M, sizeof(*A_host), y_devi, M, y_host, M);
  if(stat != CUBLASS_STATUS_SUCCESS) {
    std::err << "Data upload failed" << std::endl;
    cudaFree A_devi;
    cudaFree x_devi;
    cudaFree y_devi;
    cublasDestroy(handle);
    return EXIT_FAILURE;
  }
  

  cudaFree A_devi;
  cudaFree x_devi;
  cudaFree y_devi;
  cublasDestroy(handle);
  for(int i=0; i<M; i++) {
    std::cout << y_host[IDX2C] << " ";
  }
  std::cout << std::endl;

  free(A_host);
  free(x_host);
  free(y_host);

  return EXIT_SUCCESS;
}
