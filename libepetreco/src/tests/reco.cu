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

/** @brief System matrix calculation.
 * @param aEcsrCnlPtr_devi Array of effective row pointers. Part of ECSR sparse
 * representation of the system matrix. Representation on device.
 * @param aVxlId_devi Array of system matrix column ids (which are voxel ids).
 * Part of COO representation of the system matrix. Representation on device.
 * @param aVal_devi Array of system matrix values. Part of sparse
 * representation of the system matrix. Representation on device.
 * @param nnz_devi After execution finished: Number of non-zeros in system
 * matrix plus initial value. Should usually be initialized to 0 prior to call
 * to this function. Representation on device.
 * @param aCnlId_devi Array of system matrix row ids (which are channel ids).
 * Part of COO representation of the system matrix - is only an
 * intermediate result. Representation on device.
 * @param aCsrCnlPtr_devi Array of row Pointers. Part of CSR representation of
 * the system matrix - is only an intermediate result. Representation on device.
 * @param yRowId_devi Array of measurement vector elements' row indices. Part
 * of sparse representation of the vector. Representation on device.
 * @param effM Number of non-zero elements in measurement vector y.
 * @param handle Handle to cuSPARSE library context. */
template<typename T>
void systemMatrixCalculation(
      int * const aEcsrCnlPtr_devi, int * const aVxlId_devi, T * const aVal_devi,
      int * const nnz_devi,
      int * const aCnlId_devi, int * const aCsrCnlPtr_devi,
      int * const yRowId_devi, int * const effM_devi, int * const effM_host,
      cusparseHandle_t const & handle) {
  /* Run kernel */
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

  /* Copy nnz back to host */
  int nnz_host[1];
  HANDLE_ERROR(
        cudaMemcpy(nnz_host, nnz_devi, sizeof(nnz_host[0]), cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaDeviceSynchronize());

  /* Sort system matrix elements according to row major format */
  cooSort<val_t>(aVal_devi, aCnlId_devi, aVxlId_devi, *nnz_host);
  HANDLE_ERROR(cudaDeviceSynchronize());

  /* COO -> CSR */
  convertCoo2Csr(aCsrCnlPtr_devi, aCnlId_devi, handle, *nnz_host, NCHANNELS);
  HANDLE_ERROR(cudaDeviceSynchronize());

  /* CSR -> ECSR */
  convertCsr2Ecsr(aEcsrCnlPtr_devi, yRowId_devi, effM_host[0], aCsrCnlPtr_devi, NCHANNELS);
  HANDLE_ERROR(cudaDeviceSynchronize());
}

/** @brief Wrapper function for device memory allocation.
 * @tparam T Type of memory.
 * @param devi Pointer to allocate memory for.
 * @param n Number of elements to allocate memory for. */
template<typename T>
cudaError_t malloc_devi(T * devi, int const n) {
  return cudaMalloc((void**)&devi, sizeof(devi[0]) * n);
}

/** @brief Wrapper function for memcpy from host to device. 
 * @tparam T Type of memory.
 * @param devi Target memory on device.
 * @param host Source memory on host.
 * @param n Number of elements of type T that are copied. */
template<typename T>
cudaError_t memcpyH2D(T * const devi, T const * const host, int const n) {
  return cudaMemcpy(devi, host, sizeof(devi[0]) * n, cudaMemcpyHostToDevice);
}

/** @brief Wrapper function for memcpy from device to host.
 * @tparam T Type of memory.
 * @param host Target memory on host.
 * @param devi Source memory on device.
 * @param n Number of elements of type T that are copied. */
template<typename T>
cudaError_t memcpyD2H(T * const host, T const * const devi, int const n) {
  return cudaMemcpy(host, devi, sizeof(host[0]) * n, cudaMemcpyDeviceToHost);
}

/** @brief Find number of non-zeros in an array.
 * @tparam Type of elements.
 * @param mem Array of length n.
 * @param n Length of the array. */
int findNnz(val_t const * const mem, int const n) {
  int tmp(0);
  for(int i=0; i<n; i++) {
    if(mem[i] != 0) tmp++;
  }
  return tmp;
}
  
/** @brief Convert a dense vector into a sparse vector.
 * @tparam T Type of elements.
 * @param vctId Array that holds the vector elements' indices after function
 * returns. Must be [number of non-zeros in vct] long.
 * @param vctVal Array that holds the vector elements' values after function
 * returns. Must be [number of non-zeros in vct] long.
 * @param vct Dense vector of length n to convert.
 * @param n Length of dense vector to convert. */
template<typename T>
void makeSparseVct(int * const vctId, T * const vctVal,
      T const * const vct, int const n) {
  int id(0);
  for(int i=0; i<n; i++) {
    if(vct[i] != 0) {
      vctId[id]  = i;
      vctVal[id] = vct[i];
      id++;
    }
  }
}

/** @brief Wrapper function for device memory allocation for a sparse vector.
 * @tparam T Type of elements.
 * @param vctId_devi Array of vector elements' indices of length vctNnz. 
 * @param vctVal_devi Array of vector elements' values of length vctNnz. 
 * @param vctNnz Number of non-zeros in the vector. 
 * @return Error code of last operation in function body. */
template<typename T>
cudaError_t mallocSparseVct_devi(int * vctId_devi, T * vctVal_devi,
      int * vctNnz_devi, int const vctNnz) {
  malloc_devi<int>(vctId_devi,  vctNnz);
  malloc_devi<T>  (vctVal_devi, vctNnz);
  return malloc_devi<int>(vctNnz_devi, 1);
}

/** @brief Wrapper function for copying a sparse vector from host to device. */
template<typename T>
cudaError_t cpySparseVctH2D(
      int * const vctId_devi, T * const vctVal_devi, int * const vctNnz_devi,
      int const * const vctId_host, T const * const vctVal_host,
      int const * const vctNnz_host) {
  memcpyH2D<int>(vctId_devi,  vctId_host,  vctNnz_host[0]);
  memcpyH2D<T>(  vctVal_devi, vctVal_host, vctNnz_host[0]);
  return memcpyH2D<int>(vctNnz_devi, vctNnz_host, 1);
}



int main(int argc, char** argv) {
  /* Process command line args */
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
  
  /* Create measurement setup */
  MS setup =
    MS(
      POS0X, POS1X,
      NA, N0Z, N0Y, N1Z, N1Y,
      DA, SEGX, SEGY, SEGZ);
  
  /* Create voxel grid */
  VG grid =
    VG(
      GRIDOX, GRIDOY, GRIDOZ,
      GRIDDX, GRIDDY, GRIDDZ,
      GRIDNX, GRIDNY, GRIDNZ);
  
  /* Copy setup, grid and nrays to GPU constant memory */
  HANDLE_ERROR(cudaMemcpyToSymbol(setup_const, &setup, sizeof(MS)));
  HANDLE_ERROR(cudaMemcpyToSymbol(grid_const, &grid, sizeof(grid)));
  HANDLE_ERROR(cudaMemcpyToSymbol(nrays_const, &nrays, sizeof(int)));
  
  /*****************************************************************************
   * MEASUREMENT VECTOR
   ****************************************************************************/
  
  /* Read */
  
  /** @var effM_host Number of non-zero elements in measurement vector y.
   * Representation on host. */
  int effM_host[1];
  /** @var effM_devi Representation of effM_host on device. */
  int * effM_devi = NULL;
  
  /** @var yRowId_host Array of y row indices. Part of sparse representation of
   * measurement vector y. Will have length effM_host. Representation on host. */
  int * yRowId_host = NULL;
  /** @var yRowId_devi Representation of yRowId_host on device. */
  int * yRowId_devi = NULL;
  
  /** @var yVal_host Array of y values. Part of sparse representation of
   * measurement vector y. Will have length effM_host. Representation on host. */
  val_t * yVal_host = NULL;
  /** @var yVal_devi Representation of yVal on device. */
  val_t * yVal_devi = NULL;
  
  do{
    H5Reader reader(fn);
    int fSize = reader.sizeOfFile();
    val_t * fMem = new val_t[fSize];
    reader.read(fMem);
    effM_host[0] = findNnz(fMem, fSize);
    yRowId_host = new int[  effM_host[0]];
    yVal_host   = new val_t[effM_host[0]];
    makeSparseVct(yRowId_host, yVal_host, fMem, fSize);
    delete[] fMem;
  } while(false);
  
  HANDLE_ERROR(mallocSparseVct_devi(yRowId_devi, yVal_devi, effM_devi, effM_host[0]));
  HANDLE_ERROR(cpySparseVctH2D(yRowId_devi, yVal_devi, effM_devi,
        yRowId_host, yVal_host, effM_host));
  
  int maxNnz = effM_host[0] * VGRIDSIZE;
  
  /* Prepare objects that are meaningful to the algorithm */
  
  /** @var aCnlId_devi Array of system matrix row ids (which are channel ids).
   * Part of COO representation of the system matrix - is only an
   * intermediate result. Representation on device. */
  int * aCnlId_devi = NULL;
  HANDLE_ERROR(malloc_devi(aCnlId_devi, maxNnz));
  
  /** @var aCsrCnlPtr_devi Array of row Pointers. Part of CSR representation of
   * the system matrix - is only an intermediate result. Representation on
   * device. */
  int * aCsrCnlPtr_devi = NULL;
  HANDLE_ERROR(malloc_devi(aCsrCnlPtr_devi, NCHANNELS+1));
  
  /** @var aEcsrCnlPtr_devi Array of effective row pointers. Part of ECSR sparse
   * representation of the system matrix. Representation on device. */
  int * aEcsrCnlPtr_devi = NULL;
  HANDLE_ERROR(malloc_devi(aEcsrCnlPtr_devi, effM_host[0]+1));
  
  /** @var aVxlId_devi Array of system matrix column ids (which are voxel ids).
   * Part of (COO) sparse representation of the system matrix. Representation on
   * device. */
  int * aVxlId_devi = NULL;
  HANDLE_ERROR(malloc_devi(aVxlId_devi, maxNnz));
  
  /** @var aVal_devi Array of system matrix values. Part of sparse representation 
   * of the system matrix. Representation on device. */
  val_t * aVal_devi = NULL;
  HANDLE_ERROR(malloc_devi(aVal_devi, maxNnz));
  
  /** @var Number of non-zeros in system matrix. Representation on host. */
  int nnz_host[1];

  /** @var Number of non-zeros in system matrix. Representation on device. */
  int * nnz_devi = NULL;
  
  /** @var Density guess vector x. Representation on host. */
  val_t * x_host = new val_t[VGRIDSIZE];
  for(int i=0; i<VGRIDSIZE; i++) { x_host[i] = 1.; }
  
  /** @var Density guess vector x. Representation on device. */
  val_t * x_devi = NULL;
  HANDLE_ERROR(malloc_devi(x_devi, VGRIDSIZE));
  HANDLE_ERROR(memcpyH2D(x_devi, x_host, VGRIDSIZE));
  HANDLE_ERROR(cudaDeviceSynchronize());
  do {
    val_t norm = sum<val_t>(x_devi, VGRIDSIZE);
    scales<val_t>(x_devi, 1./norm, VGRIDSIZE);
    HANDLE_ERROR(cudaDeviceSynchronize());
  } while(false);
  
  /** @var handle Handle to cusparse library context. */
  cusparseHandle_t handle = NULL;
  
  /** @var A Matrix descriptor. Used with cuSPARSE library. */
  cusparseMatDescr_t A = NULL;
  HANDLE_CUSPARSE_ERROR(cusparseCreateMatDescr(&A));
  HANDLE_CUSPARSE_ERROR(customizeMatDescr(A, handle));
  
  /** @var alpha Scalar factor. */
  val_t alpha = 1.;
  
  /** @var beta Scalar factor. */
  val_t beta  = 0.;
  
  /** @var oneVal_host Vector of ones in measurement space. Representation on
   * host. */
  val_t * oneVal_host = new val_t[effM_host[0]];
  for(int i=0; i<effM_host[0]; i++) { oneVal_host[i] = 1.; }
  
  /** @var oneVal_host Vector of ones in measurement space. Representation on
   * device. */
  val_t * oneVal_devi = NULL;
  HANDLE_ERROR(malloc_devi(oneVal_devi, effM_host[0]));
  HANDLE_ERROR(memcpyH2D(oneVal_devi, oneVal_host, effM_host[0]));
  
  /** @var yTildeVal_devi Simulated measurement vector. Representation on
   * device. */
  val_t * yTildeVal_devi = NULL;
  HANDLE_ERROR(malloc_devi(yTildeVal_devi, effM_host[0]));
  
  /** @var eVal_devi "Error" in measurement space. Representation on device. */
  val_t * eVal_devi = NULL; 
  HANDLE_ERROR(malloc_devi(eVal_devi, effM_host[0]));
  
  /** @var c_devi Correction in grid space. Representation on device. */
  val_t * c_devi = NULL;
  HANDLE_ERROR(malloc_devi(c_devi, VGRIDSIZE));
  
  /** @var s_devi Sensitivity in grid space. Representation on device. */
  val_t * s_devi = NULL;
  HANDLE_ERROR(malloc_devi(s_devi, VGRIDSIZE));
  
  /** @var xx_host Intermediate grid space vector. Representation on host. */
  val_t * xx_host = new val_t[VGRIDSIZE];
  
  /** @var xx_devi Intermediate grid space vector. Representation on device. */
  val_t * xx_devi = NULL;
  HANDLE_ERROR(malloc_devi(xx_devi, VGRIDSIZE));
  
  
  
  /* Get system matrix */
  systemMatrixCalculation<val_t> (
        aEcsrCnlPtr_devi, aVxlId_devi, aVal_devi,
        nnz_devi,
        aCnlId_devi, aCsrCnlPtr_devi,
        yRowId_devi, effM_devi, effM_host,
        handle);
  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(memcpyD2H(nnz_host, nnz_devi, 1));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  /* Matrix vector multiplication to calculate sensitivity */
  CSRmv<val_t>()(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        effM_host[0], VGRIDSIZE, *nnz_host, &alpha, A, aVal_devi, aEcsrCnlPtr_devi, aVxlId_devi,
        oneVal_devi, &beta, s_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  
  
  /* Get system matrix */
  systemMatrixCalculation<val_t> (
        aEcsrCnlPtr_devi, aVxlId_devi, aVal_devi,
        nnz_devi,
        aCnlId_devi, aCsrCnlPtr_devi,
        yRowId_devi, effM_devi, effM_host,
        handle);
  HANDLE_ERROR(cudaDeviceSynchronize());
  HANDLE_ERROR(memcpyD2H(nnz_host, nnz_devi, 1));
  HANDLE_ERROR(cudaDeviceSynchronize());

  /* Matrix vector multiplication to simulate measurement */
  CSRmv<val_t>()(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        effM_host[0], VGRIDSIZE, *nnz_host, &alpha, A, aVal_devi, aEcsrCnlPtr_devi, aVxlId_devi,
        x_devi, &beta, yTildeVal_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  /* Division to get "error" */
  divides<val_t>(eVal_devi, yVal_devi, yTildeVal_devi, effM_host[0]);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  /* Matrix vector multiplication to backproject error on grid */
  CSRmv<val_t>()(handle, CUSPARSE_OPERATION_TRANSPOSE,
        effM_host[0], VGRIDSIZE, *nnz_host, &alpha, A, aVal_devi, aEcsrCnlPtr_devi, aVxlId_devi,
        eVal_devi, &beta, c_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  /* Apply correction */
  dividesMultiplies<val_t>(xx_devi, x_devi, c_devi, s_devi, VGRIDSIZE);
  HANDLE_ERROR(cudaDeviceSynchronize());
  
//  /* Normalize */
//  val_t norm = sum<val_t>(xx_devi, VGRIDSIZE);
//  scales<val_t>(density_Guess)
  
  /* Copy back to host */
  HANDLE_ERROR(memcpyD2H(xx_host, xx_devi, VGRIDSIZE));
  HANDLE_ERROR(cudaDeviceSynchronize());
  
  /* Write to file */
  H5DensityWriter<GridAdapter<VG, val_t> > writer(on);
  GridAdapter<VG, val_t> ga(&grid);
  writer.write(xx_host, ga);
  
  /* Cleanup */
  
  return 0;
}

