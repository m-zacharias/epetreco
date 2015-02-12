/**
 * @file reco.cu
 */
/* Author: malte
 *
 * Created on 6. Februar 2015, 17:09
 */

#include <cstdlib>

#include "CUDA_HandleError.hpp"
#include "typedefs.hpp"
#include "device_constant_memory.hpp"
#include "voxelgrid64_defines.h"
#include "real_measurementsetup_defines.h"
#include "H5Reader.hpp"
#include "getSystemMatrixDeviceOnly.cu"
#include "cooSort.hpp"
#include <cusparse.h>
#include "cusparseWrapper.hpp"
#include "CUSPARSE_HandleError.hpp"
#include "convertCsr2Ecsr.hpp"
#include "csrmv.hpp"
#include "H5DensityWriter.hpp"
#include "GridAdapter.hpp"
#include "mlemOperations.hpp"
//#include "supplement_mv.hpp"
//#include "FileTalk.hpp"

#define NBLOCKS 32

int main(int argc, char** argv) {
  // Process command line args
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
  
  // Create measurement setup
  MS setup =
    MS(
      POS0X, POS1X,
      NA, N0Z, N0Y, N1Z, N1Y,
      DA, SEGX, SEGY, SEGZ);
  
  // Create voxel grid
  VG grid =
    VG(
      GRIDOX, GRIDOY, GRIDOZ,
      GRIDDX, GRIDDY, GRIDDZ,
      GRIDNX, GRIDNY, GRIDNZ);
  
  // Copy setup, grid and nrays to GPU constant memory
  HANDLE_ERROR(cudaMemcpyToSymbol(setup_const, &setup, sizeof(MS)));
  HANDLE_ERROR(cudaMemcpyToSymbol(grid_const, &grid, sizeof(grid)));
  HANDLE_ERROR(cudaMemcpyToSymbol(nrays_const, &nrays, sizeof(int)));
  
  // Read measurement
  /**
   * @var effM Number of non-zero elements in measurement vector y.
   */
  int effM;
  /**
   * @var yRowId Array of y row indices. Part of sparse representation of
   * measurement vector y. Has length effM.
   */
  int * yRowId;
  /**
   * @var yVal Array of y values. Part of sparse representation of measurement
   * vector y. Has length effM.
   */
  val_t * yVal;
  
  do{
    H5Reader reader(fn);
    int fSize = reader.sizeOfFile();
    val_t * fMem = new val_t[fSize];
    reader.read(fMem);
    effM = 0;
    for(int i=0; i<fSize; i++) {
      if(fMem[i] != 0) effM++;
    }
    yRowId = new int[  effM];
    yVal   = new val_t[effM];
    int id = 0;
    for(int i=0; i<fSize; i++) {
      if(fMem[i] != 0) {
        yVal[id] = fMem[i];
        yRowId[id] = i;
        id++;
      }
    }
    delete[] fMem;
  } while(false);
  
  // Copy measurement vector to device
  /**
   * @var effM_devi Representation of effM on device.
   */
  int * effM_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&effM_devi,   sizeof(effM_devi[0])));
  HANDLE_ERROR(
        cudaMemcpy(effM_devi, &effM,     sizeof(effM_devi[0]),
        cudaMemcpyHostToDevice));
  /**
   * @var yRowId_devi Representation of yRowId on device.
   */
  int * yRowId_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&yRowId_devi, sizeof(yRowId_devi[0]) * effM));
  HANDLE_ERROR(
        cudaMemcpy(yRowId_devi, yRowId,  sizeof(yRowId_devi[0]) * effM,
        cudaMemcpyHostToDevice));
  /**
   * @var yVal_devi Representation of yVal on device.
   */
  val_t * yVal_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&yVal_devi,   sizeof(yVal_devi[0])   * effM));
  HANDLE_ERROR(
        cudaMemcpy(yVal_devi,   yVal,    sizeof(yVal_devi[0])   * effM,
        cudaMemcpyHostToDevice));
  
  // Get system matrix
  /**
   * @var aVxld_devi Array of system matrix column ids (which are voxel ids).
   * Part of (COO) sparse representation of the system matrix. Representation on
   * device.
   */
  int * aVxlId_devi = NULL;
  /**
   * @var aVal_devi Array of system matrix values. Part of sparse representation 
   * of the system matrix. Representation on device.
   */
  val_t * aVal_devi = NULL;
  /**
   * @var Number of non-zeros in system matrix. Representation on host.
   */
  int nnz_host[1];
  /**
   * @var Number of non-zeros in system matrix. Representation on device.
   */
  int * nnz_devi = NULL;
  /**
   * @var aEcsrCnlPtr_devi Array of effective row pointers. Part of ECSR sparse
   * representation of the system matrix. Representation on device.
   */
  int * aEcsrCnlPtr_devi = NULL;
  /**
   * @var handle Handle to cusparse library context.
   */
  cusparseHandle_t handle = NULL;
  
  do {
    int * aCnlId_devi = NULL;
    
    // Allocate memory for matrix on device
    do{
      int potNnz = effM * VGRIDSIZE;
      HANDLE_ERROR(
            cudaMalloc((void**)&aCnlId_devi, sizeof(aCnlId_devi[0]) * potNnz));
      HANDLE_ERROR(
            cudaMalloc((void**)&aVxlId_devi, sizeof(aVxlId_devi[0]) * potNnz));
      HANDLE_ERROR(
            cudaMalloc((void**)&aVal_devi,   sizeof(aVal_devi[0])   * potNnz));
    } while(false);
    
    // Initialize nnz
    *nnz_host = 0;
    HANDLE_ERROR(
          cudaMalloc((void**)&nnz_devi,   sizeof(nnz_devi[0])));
    HANDLE_ERROR(
          cudaMemcpy(nnz_devi, nnz_host, sizeof(nnz_devi[0]), cudaMemcpyHostToDevice));
    
    // Make sure, copy operations have finished
    HANDLE_ERROR(
          cudaDeviceSynchronize());

    // Run kernel
    getSystemMatrix<
          val_t, VG, Idx, Idy, Idz, MS, Id0z, Id0y, Id1z, Id1y, Ida, Trafo0, Trafo1>
          <<<NBLOCKS, TPB>>>
        ( aVal_devi,
          aVxlId_devi,
          aCnlId_devi,
          yRowId_devi,
          effM_devi,
          nnz_devi);
    HANDLE_ERROR(cudaDeviceSynchronize());
    
    // Copy nnz back to host
    HANDLE_ERROR(
          cudaMemcpy(nnz_host, nnz_devi, sizeof(nnz_host[0]), cudaMemcpyDeviceToHost));

    // Sort system matrix elements according to row major format
    cooSort<val_t>(aVal_devi, aCnlId_devi, aVxlId_devi, *nnz_host);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // COO -> CSR
    int * aCsrCnlPtr_devi = NULL;
    HANDLE_ERROR(
          cudaMalloc((void**)&aCsrCnlPtr_devi, sizeof(aCsrCnlPtr_devi[0]) * (int)(NCHANNELS+1)));
    HANDLE_CUSPARSE_ERROR(
          cusparseCreate(&handle));
    convertCoo2Csr(aCsrCnlPtr_devi, aCnlId_devi, handle, *nnz_host, NCHANNELS);
    HANDLE_ERROR(cudaDeviceSynchronize());

    // CSR -> ECSR
    HANDLE_ERROR(
          cudaMalloc((void**)&aEcsrCnlPtr_devi, sizeof(aEcsrCnlPtr_devi[0]) * (effM+1)));
    convertCsr2Ecsr(aEcsrCnlPtr_devi, yRowId_devi, effM, aCsrCnlPtr_devi, NCHANNELS);
    HANDLE_ERROR(cudaDeviceSynchronize());
    
    // Cleanup
    cudaFree(aCnlId_devi);
    cudaFree(aCsrCnlPtr_devi);
  } while(false);
  
  // Prepare density guess vector
  val_t * densityGuess_host = new val_t[VGRIDSIZE];
  for(int i=0; i<VGRIDSIZE; i++) {
    densityGuess_host[i] = 1.;
  }
  val_t * densityGuess_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&densityGuess_devi, sizeof(densityGuess_devi[0]) * VGRIDSIZE));
  HANDLE_ERROR(
        cudaMemcpy(densityGuess_devi, densityGuess_host,
              sizeof(densityGuess_devi[0]) * VGRIDSIZE, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaDeviceSynchronize());
  
  do {
    val_t norm = sum<val_t>(densityGuess_devi, VGRIDSIZE);
    scales<val_t>(densityGuess_devi, 1./norm, VGRIDSIZE);
    HANDLE_ERROR(cudaDeviceSynchronize());
  } while(false);
  
  // Matrix vector multiplication to simulate measurement
  val_t * yTildeVal_devi = NULL; 
  HANDLE_ERROR(
        cudaMalloc((void**)&yTildeVal_devi, sizeof(yTildeVal_devi[0]) * effM));
  HANDLE_ERROR(cudaDeviceSynchronize());
  cusparseMatDescr_t A = NULL;
  HANDLE_CUSPARSE_ERROR(
        cusparseCreateMatDescr(&A));
  HANDLE_CUSPARSE_ERROR(
        customizeMatDescr(A, handle));
  val_t alpha = 1.;
  val_t beta  = 0.;
  CSRmv<val_t>()(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        effM, VGRIDSIZE, *nnz_host, &alpha, A, aVal_devi, aEcsrCnlPtr_devi, aVxlId_devi,
        densityGuess_devi, &beta, yTildeVal_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Division to get "error"
  val_t * eVal_devi = NULL; 
  HANDLE_ERROR(
        cudaMalloc((void**)&eVal_devi, sizeof(eVal_devi[0]) * effM));
  HANDLE_ERROR(cudaDeviceSynchronize());
  divides<val_t>(eVal_devi, yVal_devi, yTildeVal_devi, effM);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Matrix vector multiplication to backproject error on grid
  val_t * c_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&c_devi, sizeof(c_devi[0]) * VGRIDSIZE));
  HANDLE_ERROR(cudaDeviceSynchronize());
  CSRmv<val_t>()(handle, CUSPARSE_OPERATION_TRANSPOSE,
        effM, VGRIDSIZE, *nnz_host, &alpha, A, aVal_devi, aEcsrCnlPtr_devi, aVxlId_devi,
        eVal_devi, &beta, c_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Matrix vector multiplication to calculate sensitivity
  val_t * s_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&s_devi, sizeof(s_devi[0]) * VGRIDSIZE));
  val_t * oneVal_host = new val_t[effM];
  for(int i=0; i<effM; i++) {
    oneVal_host[i] = 1.;
  }
  val_t * oneVal_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&oneVal_devi, sizeof(oneVal_devi[0]) * effM));
  HANDLE_ERROR(
        cudaMemcpy(oneVal_devi, oneVal_host, sizeof(oneVal_devi[0]) * effM,
              cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaDeviceSynchronize());
  CSRmv<val_t>()(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        effM, VGRIDSIZE, *nnz_host, &alpha, A, aVal_devi, aEcsrCnlPtr_devi, aVxlId_devi,
        oneVal_devi, &beta, s_devi);
  
  // Apply correction
  val_t * xx_devi = NULL;
  HANDLE_ERROR(
        cudaMalloc((void**)&xx_devi, sizeof(xx_devi[0]) * VGRIDSIZE));
  HANDLE_ERROR(cudaDeviceSynchronize());
  dividesMultiplies<val_t>(xx_devi, densityGuess_devi, c_devi, s_devi, VGRIDSIZE);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
//  // Normalize
//  val_t norm = sum<val_t>(xx_devi, VGRIDSIZE);
//  scales<val_t>(density_Guess)
  
  // Copy back to host
  val_t * xx_host = new val_t[VGRIDSIZE];
  HANDLE_ERROR(
        cudaMemcpy(xx_host, xx_devi, sizeof(xx_host[0]) * VGRIDSIZE,
              cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  // Write to file
  H5DensityWriter<GridAdapter<VG, val_t> > writer(on);
  GridAdapter<VG, val_t> ga(&grid);
  writer.write(xx_host, ga);
  
  // Cleanup
  delete[] xx_host;
  cudaFree(xx_devi);
  delete[] oneVal_host;
  cudaFree(oneVal_devi);
  cudaFree(c_devi);
  cudaFree(s_devi);
  cudaFree(yTildeVal_devi); 
  cudaFree(eVal_devi); 
  delete[] densityGuess_host;
  
//  // Copy grid vector to host
//  val_t * x_host = new val_t[VGRIDSIZE];
//  HANDLE_ERROR(
//        cudaMemcpy(x_host, x_devi, sizeof(x_host[0])*VGRIDSIZE, cudaMemcpyDeviceToHost));
//  HANDLE_ERROR(
//        cudaDeviceSynchronize());
//  
  
  // ###########################################################################
  // ### DEBUG
  // ###########################################################################

  delete[] yRowId;
  delete[] yVal;
  return 0;
}

