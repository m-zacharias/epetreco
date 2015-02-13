/** @file wrappers.hpp */
/* Author: malte
 *
 * Created on 13. Februar 2015, 12:47 */

#ifndef WRAPPERS_HPP
#define	WRAPPERS_HPP

#include "CUDA_HandleError.hpp"
#include "voxelgrid64_defines.h"
#include "real_measurementsetup_defines.h"
#include "getSystemMatrixDeviceOnly.cu"
#include "cooSort.hpp"
#include <cusparse.h>
#include "cusparseWrapper.hpp"
#include "convertCsr2Ecsr.hpp"
#include "GridAdapter.hpp"
#include "H5DensityWriter.hpp"
#include "VoxelGrid.hpp"
#include "H5Reader.hpp"
#include <string>

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
  
/** @brief From a dense vector extract the non-zeros' indices = measurement
 * list.
 * @tparam T Type of elements.
 * @param ml Array that holds the vector elements' indices after function
 * returns. Must be [number of non-zeros in vct] long.
 * @param vct Dense vector of length n to convert.
 * @param n Length of dense vector to convert. */
template<typename T>
void makeMeasList(int * const ml, T const * const vct, int const n) {
  int id(0);
  for(int i=0; i<n; i++) {
    if(vct[i] != 0) {
      ml[id]  = i;
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

/** @brief Wrapper function for device memory allocation for a measurement
 * list.
 * @param ml_devi Array of vector elements' indices of length vctNnz. 
 * @param vctNnz Number of non-zeros in the vector. 
 * @return Error code of last operation in function body. */
cudaError_t mallocMeasList_devi(int * ml_devi, int * vctNnz_devi,
      int const vctNnz) {
  malloc_devi<int>(ml_devi,  vctNnz);
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

/** @brief Wrapper function for copying a measurement list from host to
 * device. */
cudaError_t cpyMeasListH2D(
      int * const       ml_devi, int * const       mlN_devi,
      int const * const ml_host, int const * const mlN_host) {
  memcpyH2D<int>(ml_devi,  ml_host,  mlN_host[0]);
  return memcpyH2D<int>(mlN_devi, mlN_host, 1);
}

/** @brief Read a measurement vector from file.
 * @tparam T Type of elements.
 * @param vctId Array of vector elements' indices. Memory will be allocated
 * during function execution via new int[mlN]!
 * @param vctVal Array of vector elements' values. Memory will be allocated
 * during function execution via new T[mlN]!
 * @param mlN After function's return: Number of elements in measurement
 * list.
 * @param ifn Name of file to read from. */
template<typename T>
void readMeasVct_HDF5(int * vctId, T * vctVal, int * const mlN,
      std::string const & ifn) {
  H5Reader reader(ifn);
  int fN = reader.sizeOfFile();
  T * fMem = new T[fN];
  reader.read(fMem);
  mlN[0] = findNnz(fMem, fN);
  vctId  = new int[mlN[0]];
  vctVal = new T[  mlN[0]];
  makeSparseVct(vctId, vctVal, fMem, fN);
  delete[] fMem;
}

/** @brief Read a measurement list from file.
 * @tparam T Type of elements in file.
 * @param ml Measurement list. Memory for the list will be allocated during
 * function execution via new int[mlN]!
 * @param mlN After function's return: Number of elements in measurement
 * list.
 * @param ifn Name of file to read from. */
template<typename T>
void readMeasList_HDF5(int * ml, int * const mlN, std::string const & ifn) {
  H5Reader reader(ifn);
  int fN = reader.sizeOfFile();
  T * fMem = new T[fN];
  reader.read(fMem);
  mlN[0] = findNnz(fMem, fN);
  ml  = new int[mlN[0]];
  makeMeasList(ml, fMem, fN);
  delete[] fMem;
}

/** @brief Write density to hdf5.
 * @tparam T Type of density elements.
 * @param density Array of density data.
 * @param ofn Output filename. */
template<typename T>
void writeDensity_HDF5(T const * const density, std::string const ofn,
      DefaultVoxelGrid<T> const & grid) {
  H5DensityWriter<GridAdapter<DefaultVoxelGrid<T>, T> > writer(ofn);
  GridAdapter<DefaultVoxelGrid<T>, T> ga(&grid);
  writer.write(density, ga);
}
#endif	/* WRAPPERS_HPP */

