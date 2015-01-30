/* 
 * File:   example_cuda_kernel-header-main.cu
 * Author: malte
 *
 * Created on 23. Januar 2015, 12:34
 */

#include <cstdlib>
#include <iostream>
#include "example_cuda_kernel-header.hpp"
#include "CUDA_HandleError.hpp"

#define N 1000

typedef float val_t;

int main() {
  val_t * a_host      = new val_t[N];
  val_t * b_host      = new val_t[N];
  val_t * result_host = new val_t[N];
  int     n_host      = N;
  
  for(int i=0; i<N; i++) {
    a_host[i]       = i;
    b_host[i]       = N-i;
    result_host[i]  = 0;
  }
  
  val_t * a_devi      = NULL;
  val_t * b_devi      = NULL;
  val_t * result_devi = NULL;
  int * n_devi        = NULL;
  
  HANDLE_ERROR(cudaMalloc((void**)&a_devi,      sizeof(a_devi[0])      * N));
  HANDLE_ERROR(cudaMalloc((void**)&b_devi,      sizeof(b_devi[0])      * N));
  HANDLE_ERROR(cudaMalloc((void**)&result_devi, sizeof(result_devi[0]) * N));
  HANDLE_ERROR(cudaMalloc((void**)&n_devi,      sizeof(n_devi[0])));
  
  HANDLE_ERROR(cudaMemcpy(a_devi,      a_host,      sizeof(a_devi[0])      * N, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(b_devi,      b_host,      sizeof(b_devi[0])      * N, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(result_devi, result_host, sizeof(result_devi[0]) * N, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(n_devi,      &n_host,     sizeof(n_devi[0]),          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  add<<<TPB, (N+TPB-1)/TPB>>>(a_devi, b_devi, result_devi, n_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  HANDLE_ERROR(cudaMemcpy(result_host, result_devi, sizeof(result_devi[0]) * N, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  for(int i=0; i<N; i++) {
    std::cout << i;
    if(i%20 == 0) {
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;
  
  delete[] a_host;
  delete[] b_host;
  delete[] result_host;
  HANDLE_ERROR(cudaFree(a_devi));
  HANDLE_ERROR(cudaFree(b_devi));
  HANDLE_ERROR(cudaFree(result_devi));
  HANDLE_ERROR(cudaFree(n_devi));
  
  exit(EXIT_SUCCESS);
}

