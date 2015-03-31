/** @file test_getSystemMatrixDeviceOnly.cu */
/* Author: malte
 *
 * Created on 16. Januar 2015, 15:20 */

#include <cstdlib>
#include <iostream>
#include <fstream>

#define NBLOCKS 32

#include "wrappers.hpp"
#include "getSystemMatrixDeviceOnly.cu"
#include "real_measurementsetup_defines.h"
//#include "voxelgrid10_defines.h"
//#include "voxelgrid20_defines.h"
//#include "voxelgrid52_defines.h"
#include "voxelgrid64_defines.h"
#include "CUDA_HandleError.hpp"
#include "typedefs.hpp"
#include "device_constant_memory.hpp"



int main(int argc, char** argv) {
  int const nargs(3);
  if(argc!=nargs+1) {
    std::cerr << "Error: Wrong number of arguments. Exspected: "
              << nargs << ":" << std::endl
              << "  filename of measurement" << std::endl
              << "  filename of output" << std::endl
              << "  number of rays" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string const fn(argv[1]);
  std::string const on(argv[2]);
  
  
  int const nrays(atoi(argv[3]));
  HANDLE_ERROR(cudaMemcpyToSymbol(nrays_const, &nrays, sizeof(int)));

  
  MS setup = MS(POS0X, POS1X, NA, N0Z, N0Y, N1Z, N1Y, DA, SEGX, SEGY, SEGZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(setup_const, &setup, sizeof(MS)));
  
  
  VG grid = VG(GRIDOX, GRIDOY, GRIDOZ, GRIDDX, GRIDDY, GRIDDZ, GRIDNX, GRIDNY, GRIDNZ);
  HANDLE_ERROR(cudaMemcpyToSymbol(grid_const, &grid, sizeof(grid)));
  
  
  int mlSize_host[1];
  int * ml_host = NULL;
  readMeasList_HDF5<float>(ml_host, mlSize_host, fn);
  
  int * mlSize_devi = NULL;
  int * ml_devi = NULL;
  HANDLE_ERROR(mallocMeasList_devi(ml_devi, mlSize_devi, mlSize_host[0]));
  HANDLE_ERROR(cpyMeasListH2D(ml_devi, mlSize_devi, ml_host, mlSize_host));
  
  
  int const memSize = mlSize_host[0] * VGRIDSIZE;
  int * cnlId_devi = NULL;
  int * vxlId_devi = NULL;
  val_t * sme_devi = NULL;
  int truckDest_host[1] = {0};
  int * truckDest_devi = NULL;
  HANDLE_ERROR(malloc_devi(cnlId_devi, memSize));
  HANDLE_ERROR(malloc_devi(vxlId_devi, memSize));
  HANDLE_ERROR(malloc_devi(sme_devi, memSize));
  HANDLE_ERROR(malloc_devi(truckDest_devi, 1));
  
  HANDLE_ERROR(memcpyH2D(truckDest_devi, truckDest_host, 1));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  getSystemMatrix<
        val_t, VG, Idx, Idy, Idz, MS, Id0z, Id0y, Id1z, Id1y, Ida, Trafo0, Trafo1>
        <<<NBLOCKS, TPB>>>
      ( sme_devi, vxlId_devi, cnlId_devi, ml_devi, mlSize_devi, truckDest_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  HANDLE_ERROR(memcpyD2H(truckDest_host, truckDest_devi, 1));
  
  std::vector<val_t> sme_host(truckDest_host[0], 0.);
  HANDLE_ERROR(memcpyD2H(&(*sme_host.begin()), sme_devi, truckDest_host[0]))
  std::stable_sort(sme_host.begin(), sme_host.end());
  
  
  val_t sum(0);
  for(int i=0; i<truckDest_host[0]; i++) { sum += sme_host[i]; }
  
  std::ofstream out(on.c_str());
  if(!out) {
    std::cerr << __FILE__ << "(" << __LINE__ << "): Error: Could not open "
              << on << " for writing." << std::endl;
    exit(EXIT_FAILURE);
  }
  out << sum;
  
  
  if(sum != 0.) exit(EXIT_SUCCESS);
  exit(EXIT_FAILURE);
}

