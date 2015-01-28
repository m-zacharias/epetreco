/** @file test_getSystemMatrixDeviceOnly.cu */
/* 
 * File:   test_getSystemMatrixDeviceOnly.cu
 * Author: malte
 *
 * Created on 16. Januar 2015, 15:20
 */

#include <stdlib.h>
#include "FileTalk.hpp"
#include "getSystemMatrixDeviceOnly.cu"
#include "VoxelGrid.hpp"
#include "MeasurementSetup.hpp"
#include "MeasurementSetupLinIndex.hpp"
#include "MeasurementSetupTrafo2CartCoord.hpp"
#include "H5File2DefaultMeasurementList.h"
#include "H5DensityWriter.hpp"
#include "real_measurementsetup_defines.h"
//#include "voxelgrid10_defines.h"
//#include "voxelgrid20_defines.h"
//#include "voxelgrid52_defines.h"
#include "voxelgrid64_defines.h"
#include <iostream>
#include "CUDA_HandleError.hpp"

#include "typedefs.hpp"
#include "device_constant_memory.hpp"

/*
 * Simple C++ Test Suite
 */



#define NBLOCKS 32

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



template<typename T>
struct SparseEntry {
  int _vxlId, _cnlId;
  T _sme;
  
  SparseEntry()
  : _vxlId(0), _cnlId(0), _sme(0.) {}
  
  SparseEntry( int const & vxlId, int const & cnlId, T const & sme )
  : _vxlId(vxlId), _cnlId(cnlId), _sme(sme) {}
  
  SparseEntry( SparseEntry const & ori )
  : _vxlId(ori._vxlId), _cnlId(ori._cnlId), _sme(ori._sme) {}
  
  void operator= ( SparseEntry const rhs ) {
    this->_vxlId = rhs._vxlId;
    this->_cnlId = rhs._cnlId;
    this->_sme   = rhs._sme;
  }
};

template<typename T>
struct SmallerVxlId {
  bool operator() ( T const & lhs, T const & rhs ) const {
    return (lhs._vxlId < rhs._vxlId);
  }
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
  
  HANDLE_ERROR(cudaMemcpyToSymbol(nrays_const, &nrays, sizeof(int)));

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
  
  // Copy channel ids to plain array
  int const mlSize_host = list.size();
  int * ml_host = new int[mlSize_host];
  for(int i=0; i<mlSize_host; i++) {
    ml_host[i] = list.cnlId(i);
  }
  
  // Allocate memory for measurement list array and size
  int * mlSize_devi = NULL;
  int * ml_devi = NULL;
  HANDLE_ERROR(cudaMalloc((void**)&mlSize_devi, sizeof(mlSize_devi[0])));
  HANDLE_ERROR(cudaMalloc((void**)&ml_devi, sizeof(ml_devi[0]) *mlSize_host));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Copy measurement list to device
  HANDLE_ERROR(
        cudaMemcpy(ml_devi, ml_host, sizeof(ml_devi[0]) *mlSize_host, cudaMemcpyHostToDevice));
  HANDLE_ERROR(
        cudaMemcpy(mlSize_devi, &mlSize_host, sizeof(mlSize_devi[0]), cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Memory size for sparse matrix
  int const memSize = list.size() * grid.gridnx() * grid.gridny() * grid.gridnz();
  
  // Allocate memory for sparse matrix (=workqueue + matrix values) on device
  int * cnlId_devi = NULL;
  int * vxlId_devi = NULL;
  val_t * sme_devi = NULL;
  int * truckDest_devi = NULL;
  int truckDest_host = 0;
  HANDLE_ERROR(cudaMalloc((void**)&cnlId_devi, sizeof(cnlId_devi[0]) *memSize));
  HANDLE_ERROR(cudaMalloc((void**)&vxlId_devi, sizeof(vxlId_devi[0]) *memSize));
  HANDLE_ERROR(cudaMalloc((void**)&sme_devi,   sizeof(sme_devi[0])   *memSize));
  HANDLE_ERROR(cudaMalloc((void**)&truckDest_devi, sizeof(truckDest_devi[0])));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
//  // Copy Workqueue to device
//  SAYLINE(__LINE__-1);
//  HANDLE_ERROR(cudaMemcpy(
//        wqCnlId_devi, &(*wqCnlId_host.begin()), sizeof(wqCnlId_devi[0]) *nFound, cudaMemcpyHostToDevice));
//  HANDLE_ERROR(cudaMemcpy(
//        wqVxlId_devi, &(*wqVxlId_host.begin()), sizeof(wqVxlId_devi[0]) *nFound, cudaMemcpyHostToDevice));
//  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Copy to device
  HANDLE_ERROR(cudaMemcpy(
        truckDest_devi, &truckDest_host, sizeof(truckDest_devi[0]), cudaMemcpyHostToDevice));
  
  // Kernel launch
  SAYLINE(__LINE__-1);
  getSystemMatrix<
        val_t, VG, Idx, Idy, Idz, MS, Id0z, Id0y, Id1z, Id1y, Ida, Trafo0, Trafo1>
        <<<NBLOCKS, TPB>>>
      ( sme_devi,
        vxlId_devi,
        cnlId_devi,
        ml_devi,
        mlSize_devi,
        truckDest_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Copy number of elements to host
  SAYLINE(__LINE__-1);
  int nFound(0);
  HANDLE_ERROR(cudaGetLastError());
  HANDLE_ERROR(cudaMemcpy(
        &nFound, truckDest_devi, sizeof(nFound), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Allocate memory for sparse matrix on host
  SAYLINE(__LINE__-1);
  std::vector<val_t> sme_host(nFound, 0.);
  std::vector<int> vxlId_host(nFound, 0);
  std::vector<int> cnlId_host(nFound, 0);
  
  // Copy sparse matrix to host
  SAYLINE(__LINE__-1);
  HANDLE_ERROR(cudaMemcpy(
        &(*sme_host.begin()),   sme_devi,   sizeof(sme_host[0]  ) * nFound, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(
        &(*vxlId_host.begin()), vxlId_devi, sizeof(vxlId_host[0]) * nFound, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(
        &(*cnlId_host.begin()), cnlId_devi, sizeof(cnlId_host[0]) * nFound, cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Transform sparse matrix, sort by vxlId
  SAYLINE(__LINE__-1);
  std::vector< SparseEntry<val_t> > matrix(nFound, SparseEntry<val_t>(0, 0, 0.));
  for(int memId=0; memId<vxlId_host.size(); memId++) {
    matrix[memId] = SparseEntry<val_t>(vxlId_host[memId], cnlId_host[memId], sme_host[memId]);
  }
  std::stable_sort(matrix.begin(), matrix.end(), SmallerVxlId< SparseEntry<val_t> >());
  
  // Sum up values
  SAYLINE(__LINE__-1);
  val_t sum(0);
  for(int i=0; i<nFound; i++) {
    sum += matrix[i]._sme;
  }
  std::cout << "Sum is: " << sum << std::endl;
  
  
  // Create grid memory for backprojection
  SAYLINE(__LINE__-1);
  int const gridsize(grid.gridnx()*grid.gridny()*grid.gridnz());
  val_t * mem = new val_t[gridsize];
  for(int vxlId=0; vxlId<gridsize; vxlId++) {
    mem[vxlId] = 0.;
  }
  
  // Backproject "workqueue" on grid
  SAYLINE(__LINE__-1);
  for(int wqId=0; wqId<nFound; wqId++) {
    int const vxlId =  matrix[wqId]._vxlId;
    std::cout << "vxlId: " << vxlId << ", gridsize: " << gridsize << std::endl;
    mem[vxlId]      += matrix[wqId]._sme;
  }
  
  // Write to hdf5
  SAYLINE(__LINE__-1);
  H5DensityWriter<GridAdapter<val_t> > writer(on);
  GridAdapter<val_t> ga(&grid);
  writer.write(mem, ga);
  
  return (EXIT_SUCCESS);
}

