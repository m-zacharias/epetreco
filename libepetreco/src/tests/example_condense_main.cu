/* 
 * File:   example_condense_main.cu
 * Author: malte
 *
 * Created on 26. November 2014, 16:15
 */

#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include "CUDA_HandleError.hpp"
#include "example_condense.h"

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
  int const nargs(1);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected " << nargs
              << ":" << std::endl
              << "    output filename" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string out_fn(argv[1]);
  
  std::vector<val_t> passed_host(SIZE, 0.);
  std::vector<int>   stuff_host(SIZE, 0);
  std::vector<int>   block_host(SIZE, 0);
  int   memId_host[1] = {0};
  
  val_t * passed_devi = NULL;
  int *   stuff_devi  = NULL;
  int *   block_devi  = NULL;
  int *   memId_devi  = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&passed_devi, sizeof(passed_devi[0]) * SIZE));
  HANDLE_ERROR(cudaMalloc((void**)&stuff_devi,  sizeof(stuff_devi[0])  * SIZE));
  HANDLE_ERROR(cudaMalloc((void**)&block_devi,  sizeof(block_devi[0])  * SIZE));
  HANDLE_ERROR(cudaMalloc((void**)&memId_devi,  sizeof(memId_devi[0])));
  HANDLE_ERROR(cudaMemcpy(passed_devi, &passed_host[0], sizeof(passed_devi[0]) * SIZE, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(stuff_devi,  &stuff_host[0],  sizeof(stuff_devi[0])  * SIZE, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(block_devi,  &block_host[0],  sizeof(block_devi[0])  * SIZE, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(memId_devi,  &memId_host[0],  sizeof(memId_devi[0]),         cudaMemcpyHostToDevice));
  
  condense<<<NBLOCKS, TPB>>>(passed_devi, stuff_devi, block_devi, memId_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  HANDLE_ERROR(cudaMemcpy(&passed_host[0], passed_devi, sizeof(passed_host[0]) * SIZE, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&stuff_host[0],  stuff_devi,  sizeof(stuff_host[0])  * SIZE, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&block_host[0],  block_devi,  sizeof(block_host[0])  * SIZE, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(&memId_host[0],  memId_devi,  sizeof(memId_host[0]),         cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
//  std::cout << "Found: " << *memId_host << std::endl;
//  for(int i=0; i<*memId_host; i++) {
//    std::cout << "passed[" << i 
//              << "]: " << passed_host[i]
//              << ", stuff: " << stuff_host[i]
//              << ", block: " << block_host[i]
//              << std::endl;
//  }
  
  std::sort(passed_host.begin(), passed_host.end());
  std::ofstream out(out_fn.c_str(), std::ofstream::trunc);
  if(!out.is_open()) {
    std::cerr << "Error: Could not open file " << out_fn << std::endl;
    
    cudaFree(passed_devi);
    cudaFree(stuff_devi);
    cudaFree(block_devi);
    cudaFree(memId_devi);
    
    exit(EXIT_FAILURE);
  }
  
  for(int i=0; i<passed_host.size(); i++) {
    out << passed_host[i] << std::endl;
  }
  
  cudaFree(passed_devi);
  cudaFree(stuff_devi);
  cudaFree(block_devi);
  cudaFree(memId_devi);
  
  return 0;
}

