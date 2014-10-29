/* 
 * File:   test_getSystemMatrixElement.cu
 * Author: malte
 *
 * Created on 22. Oktober 2014, 10:29
 */

#include <stdlib.h>
#include "FileTalk.hpp"
#include "getSystemMatrixElement.cu"
#include "getWorkqueue.hpp"
#include "VoxelGrid.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementSetupLinIndex.hpp"
#include "MeasurementSetupTrafo2CartCoord.hpp"
#include "H5File2DefaultMeasurementList.h"
#include "H5DensityWriter.hpp"
#include "real_measurementsetup_defines.h"
//#include "voxelgrid10_defines.h"
#include "voxelgrid52_defines.h"
#include <iostream>
#include "CUDA_HandleError.hpp"

#include "typedefs.hpp"
#include "device_constant_memory.hpp"

/*
 * Simple C++ Test Suite
 */

#define TPB 256
#define QUEUELENGTH 1024*1024*10

template<typename T>
class GridAdapter {
public:
  GridAdapter(VG * grid) {
    _grid = grid;
  }
  
  void getOrigin( T * const origin ) const {
    origin[0] = _grid->gridox();
    origin[1] = _grid->gridoy();
    origin[2] = _grid->gridoz();
  }
  
  void getVoxelSize( T * const voxelSize ) const {
    voxelSize[0] = _grid->griddx();
    voxelSize[1] = _grid->griddy();
    voxelSize[2] = _grid->griddz();
  }
  
  void getNumberOfVoxels( int * const number ) const {
    number[0] = _grid->gridnx();
    number[1] = _grid->gridny();
    number[2] = _grid->gridnz();
  }
  
private:
  VG * _grid;
};



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

  MS setup =
    MS(
      POS0X, POS1X,
      NA, N0Z, N0Y, N1Z, N1Y,
      DA, SEGX, SEGY, SEGZ);
  
  HANDLE_ERROR(cudaMemcpyToSymbol(setup_const, &setup, sizeof(MS)));
  
  VG grid =
    VG(
      GRIDOX, GRIDOY, GRIDOZ,
      GRIDDX, GRIDDY, GRIDDZ,
      GRIDNX, GRIDNY, GRIDNZ);
  
  HANDLE_ERROR(cudaMemcpyToSymbol(grid_const, &grid, sizeof(grid)));
  
  ML list =
    H5File2DefaultMeasurementList<val_t>(fn, NA*N0Z*N0Y*N1Z*N1Y);
  
  // Allocate memory for workqueue and matrix values, i.e. sparse matrix
  SAYLINE(__LINE__-1);
  int * wqCnlId_host = new int[QUEUELENGTH];
  int * wqVxlId_host = new int[QUEUELENGTH];
  val_t *   val_host = new val_t[  QUEUELENGTH];
  
  int * wqCnlId_devi = NULL;
  int * wqVxlId_devi = NULL;
  val_t *   val_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&wqCnlId_devi, sizeof(wqCnlId_devi[0]) *QUEUELENGTH));
  HANDLE_ERROR(cudaMalloc((void**)&wqVxlId_devi, sizeof(wqVxlId_devi[0]) *QUEUELENGTH));
  HANDLE_ERROR(cudaMalloc((void**)&val_devi,     sizeof(val_devi[0])     *QUEUELENGTH));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Get Workqueue
  SAYLINE(__LINE__-1);
  int listId(0); int vxlId(0);
  int nFound =
    getWorkqueueEntries<
          val_t,
          ML, VG, MS,
          Id0z, Id0y, Id1z, Id1y, Ida,
          Trafo0, Trafo1> (
          QUEUELENGTH, wqCnlId_host, wqVxlId_host, listId, vxlId, &list, &grid, &setup);
  
  // Copy Workqueue to device
  SAYLINE(__LINE__-1);
  HANDLE_ERROR(cudaMemcpy(
        wqCnlId_devi, wqCnlId_host, sizeof(wqCnlId_devi[0]) *QUEUELENGTH, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(
        wqVxlId_devi, wqVxlId_host, sizeof(wqVxlId_devi[0]) *QUEUELENGTH, cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Kernel launch
  SAYLINE(__LINE__-1);
  calcSystemMatrixElement<
        val_t, VG, Idx, Idy, Idz, MS, Id0z, Id0y, Id1z, Id1y, Ida, Trafo0, Trafo1>
        <<<(QUEUELENGTH+TPB-1)/TPB, TPB>>> (
        wqCnlId_devi, wqVxlId_devi, val_devi, QUEUELENGTH, nrays);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Copy matrix values to host
  SAYLINE(__LINE__-1);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaMemcpy(
        val_host, val_devi, sizeof(val_host[0]) * QUEUELENGTH, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Sum up values
  SAYLINE(__LINE__-1);
  val_t sum(0);
  for(int i=0; i<QUEUELENGTH; i++) {
    sum += val_host[i];
  }
  std::cout << "Sum is: " << sum << std::endl;
  
  
  // Create grid memory for backprojection
  SAYLINE(__LINE__-1);
  int const gridsize(grid.gridnx()*grid.gridny()*grid.gridnz());
  val_t * mem = new val_t[gridsize];
  for(int vxlId=0; vxlId<gridsize; vxlId++) {
    mem[vxlId] = 0.;
  }
  
  // Backproject workqueue on grid
  SAYLINE(__LINE__-1);
  for(int wqId=0; wqId<nFound; wqId++) {
    int vxlId   = wqVxlId_host[wqId];
    mem[vxlId] += val_host[    wqId];
  }
  
  // Write to hdf5
  SAYLINE(__LINE__-1);
  H5DensityWriter<GridAdapter<val_t> > writer(on);
  GridAdapter<val_t> ga(&grid);
  writer.write(mem, ga);
  
  return (EXIT_SUCCESS);
}
