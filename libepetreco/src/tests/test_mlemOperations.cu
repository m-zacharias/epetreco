/**
 * @file test_mlemOperations.cu
 */
/* 
 * Author: malte
 *
 * Created on 4. Februar 2015, 14:15
 */

#include <iostream>
#include "CUDA_HandleError.hpp"
#include "mlemOperations.hpp"

#define N 10

typedef float val_t;

int main(int argc, char** argv) {
  // Create host arrays
  val_t A_host[N];
  val_t B_host[N];
  val_t C_host[N];
  val_t D_host[N];
  
  // Fill host arrays
  for(int i=0; i<N; i++) {
    A_host[i] = (i+1)*(i+1);
    C_host[i] = i+1;
    B_host[i] = 10;
    D_host[i] = 0;
  }
  
  // Create device arrays
  val_t * A_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&A_devi, sizeof(A_devi[0]) * N));
  val_t * B_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&B_devi, sizeof(B_devi[0]) * N));
  val_t * C_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&C_devi, sizeof(C_devi[0]) * N));
  val_t * D_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&D_devi, sizeof(D_devi[0]) * N));
  
  // Copy to device arrays
  HANDLE_ERROR(
        cudaMemcpy(A_devi, A_host, sizeof(A_devi[0]) * N, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
        cudaMemcpy(B_devi, B_host, sizeof(B_devi[0]) * N, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
        cudaMemcpy(C_devi, C_host, sizeof(C_devi[0]) * N, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
        cudaMemcpy(D_devi, D_host, sizeof(D_devi[0]) * N, cudaMemcpyHostToDevice));
  
  // Divides
  divides<val_t>(D_devi, A_devi, C_devi, N);
  HANDLE_ERROR(
        cudaDeviceSynchronize());
  
  // Copy result to host
  HANDLE_ERROR(
        cudaMemcpy(D_host, D_devi, sizeof(D_host[0]) * N, cudaMemcpyDeviceToHost));
  
  // Show results
  std::cout << "D = A / C = " << std::endl;
  for(int i=0; i<N; i++) {
    std::cout << D_host[i] << std::endl;
  }
  std::cout << std::endl;
  
  // Divides multiplies
  dividesMultiplies<val_t>(D_devi, A_devi, B_devi, C_devi, N);
  HANDLE_ERROR(
        cudaDeviceSynchronize());
  
  // Copy result to host
  HANDLE_ERROR(
        cudaMemcpy(D_host, D_devi, sizeof(D_host[0]) * N, cudaMemcpyDeviceToHost));
  
  // Show results
  std::cout << "D = A / B * C = " << std::endl;
  for(int i=0; i<N; i++) {
    std::cout << D_host[i] << std::endl;
  }
  std::cout << std::endl;

  // Sum
  val_t norm = sum<val_t>(D_devi, N);
  HANDLE_ERROR(
        cudaDeviceSynchronize());
  
  // Show results
  std::cout << "norm =" << std::endl;
  std::cout << norm << std::endl;
  std::cout << std::endl;
    
  // Scales
  scales<val_t>(D_devi, (1./norm), N);
  
  // Copy result to host
  HANDLE_ERROR(
        cudaMemcpy(D_host, D_devi, sizeof(D_host[0]) * N, cudaMemcpyDeviceToHost));
  
  // Show results
  std::cout << "D = D * " << 1./norm << " = " << std::endl;
  for(int i=0; i<N; i++) {
    std::cout << D_host[i] << std::endl;
  }
  std::cout << std::endl;
  
  cudaFree(A_devi);
  cudaFree(B_devi);
  cudaFree(C_devi);
  cudaFree(D_devi);
  
  return 0;
}

