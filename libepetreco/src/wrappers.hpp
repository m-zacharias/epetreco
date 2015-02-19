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
#include <vector>

#ifdef MEASURE_TIME
#include <ctime>
#include <iostream>
#include <string>
void printTimeDiff(clock_t const end, clock_t const beg,
      std::string const & mes) {
  std::cout << mes << (float(end-beg)/CLOCKS_PER_SEC) << " s" << std::endl;
}
#endif /* MEASURE_TIME */


/** @brief Wrapper function for device memory allocation.
 * @tparam T Type of memory.
 * @param devi Pointer to allocate memory for.
 * @param n Number of elements to allocate memory for. */
template<typename T>
cudaError_t malloc_devi(T * & devi, int const n) {
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

/** @brief Wrapper function for device memory allocation for a sparse vector.
 * @tparam T Type of elements.
 * @param vctId_devi Array of vector elements' indices of length vctNnz.
 * @param vctVal_devi Array of vector elements' values of length vctNnz.
 * @param vctNnz Number of non-zeros in the vector. 
 * @return Error code of last operation in function body. */
template<typename T>
cudaError_t mallocSparseVct_devi(int * & vctId_devi, T * & vctVal_devi,
      int const vctNnz) {
         malloc_devi<int>(vctId_devi,  vctNnz);
  return malloc_devi<T>  (vctVal_devi, vctNnz);
}

/** @brief Wrapper function for device memory allocation for a measurement
 * list.
 * @param ml_devi Array of measurement list entries of length mlN. 
 * @param mlN Number of entries in the measurement list. 
 * @return Error code of last operation in function body. */
cudaError_t mallocMeasList_devi(int * & ml_devi, int const mlN) {
  return malloc_devi<int>(ml_devi,  mlN);
}

template<typename T>
cudaError_t mallocSystemMatrix_devi(int * & mtxCnlId_devi,
      int * & mtxCsrCnlPtr_devi, int * & mtxEcsrCnlPtr_devi,
      int * & mtxVxlId_devi, T * & mtxVal_devi, int const nRows,
      int const nMemRows, int const nCols) {
         malloc_devi<int>(  mtxCnlId_devi,      (nMemRows*nCols));
         malloc_devi<int>(  mtxCsrCnlPtr_devi,  (nRows+1));
         malloc_devi<int>(  mtxEcsrCnlPtr_devi, (nMemRows+1));
         malloc_devi<int>(  mtxVxlId_devi,      (nMemRows*nCols));
  return malloc_devi<val_t>(mtxVal_devi,        (nMemRows*nCols));
}

/** @brief Wrapper function for copying a sparse vector from host to device.
 * @tparam T Type of vector elements.
 * @param vctId_devi Array of vector elements' indices of length vctNnz.
 * Representation on device.
 * @param vctVal_devi Array of vector elements' values of length vctNnz.
 * Representation on device.
 * @param vctId_host Array of vector elements' indices of length vctNnz.
 * Representation on host.
 * @param vctVal_host Array of vector elements' values of length vctNnz.
 * Representation on host.
 * @param vctNnz Number of non-zeros in the vector. */
template<typename T>
cudaError_t cpySparseVctH2D(
      int * const       vctId_devi, T * const       vctVal_devi,
      int const * const vctId_host, T const * const vctVal_host,
      int const vctNnz) {
         memcpyH2D<int>(vctId_devi,  vctId_host,  vctNnz);
  return memcpyH2D<T>(  vctVal_devi, vctVal_host, vctNnz);
}

/** @brief Wrapper function for copying a measurement list from host to device.
 * @param ml_devi Array of measurement list entries of length mlN.
 * Representation on device.
 * @param ml_host Array of measurement list entries of length mlN.
 * Representation on host.
 * @param mlN Number of entries in the measurement list. */
cudaError_t cpyMeasListH2D(
      int * const       ml_devi, int const * const ml_host,
      int const mlN) {
  return memcpyH2D<int>(ml_devi,  ml_host,  mlN);
}

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
 * @param nY_host Treat first nY_host elements listed in yRowId_devi.
 * Representation on host.
 * @param handle Handle to cuSPARSE library context. */
template<typename T>
void systemMatrixCalculation(
      int * const aEcsrCnlPtr_devi, int * const aVxlId_devi, T * const aVal_devi,
      int * const nnz_devi,
      int * const aCnlId_devi, int * const aCsrCnlPtr_devi,
      int * const yRowId_devi, int const * const & nY_host,
      cusparseHandle_t const & handle) {
#ifdef MEASURE_TIME
  clock_t time1 = clock();
#endif /* MEASURE_TIME */

  /* Copy to device */
  int * nY_devi = NULL;
  HANDLE_ERROR(malloc_devi<int>(nY_devi, 1));
  HANDLE_ERROR(memcpyH2D<int>(nY_devi, nY_host, 1));
  HANDLE_ERROR(cudaDeviceSynchronize());
#ifdef MEASURE_TIME
  clock_t time2 = clock();
  printTimeDiff(time2, time1, "systemMatrixCalculation: Copy nY to device: ");
#endif /* MEASURE_TIME */
  
  /* Run kernel */
  getSystemMatrix<
        val_t, VG, Idx, Idy, Idz, MS, Id0z, Id0y, Id1z, Id1y, Ida, Trafo0_inplace, Trafo1_inplace>
        <<<NBLOCKS, TPB>>>
      ( aVal_devi,
        aVxlId_devi,
        aCnlId_devi,
        yRowId_devi,
        nY_devi,
        nnz_devi);
  HANDLE_ERROR(cudaDeviceSynchronize());
#ifdef MEASURE_TIME
  clock_t time3 = clock();
  printTimeDiff(time3, time2, "systemMatrixCalculation: Run kernel: ");
#endif /* MEASURE_TIME */

  /* Copy nnz back to host */
  int nnz_host[1];
  HANDLE_ERROR(memcpyD2H<int>(nnz_host, nnz_devi, 1));
  HANDLE_ERROR(cudaDeviceSynchronize());
#ifdef MEASURE_TIME
  clock_t time4 = clock();
  printTimeDiff(time4, time3, "systemMatrixCalculation: Copy nnz to host: ");
#endif /* MEASURE_TIME */

  /* Sort system matrix elements according to row major format */
  cooSort<val_t>(aVal_devi, aCnlId_devi, aVxlId_devi, *nnz_host);
  HANDLE_ERROR(cudaDeviceSynchronize());
#ifdef MEASURE_TIME
  clock_t time5 = clock();
  printTimeDiff(time5, time4, "systemMatrixCalculation: Sort elems: ");
#endif /* MEASURE_TIME */

  /* COO -> CSR */
  convertCoo2Csr(aCsrCnlPtr_devi, aCnlId_devi, handle, *nnz_host, NCHANNELS);
  HANDLE_ERROR(cudaDeviceSynchronize());
#ifdef MEASURE_TIME
  clock_t time6 = clock();
  printTimeDiff(time6, time5, "systemMatrixCalculation: COO -> CSR: ");
#endif /* MEASURE_TIME */

  /* CSR -> ECSR */
  convertCsr2Ecsr(aEcsrCnlPtr_devi, yRowId_devi, nY_host[0], aCsrCnlPtr_devi, NCHANNELS);
  HANDLE_ERROR(cudaDeviceSynchronize());
#ifdef MEASURE_TIME
  clock_t time7 = clock();
  printTimeDiff(time7, time6, "systemMatrixCalculation: CSR -> ECSR: ");
#endif /* MEASURE_TIME */
  
  /* Cleanup */
  cudaFree(nY_devi);
#ifdef MEASURE_TIME
  clock_t time8 = clock();
  printTimeDiff(time8, time7, "systemMatrixCalculation: Cleanup: ");
#endif /* MEASURE_TIME */
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

/** @brief Read a measurement vector from file.
 * @tparam T Type of elements.
 * @param vctId Array of vector elements' indices. Memory will be allocated
 * during function execution via new int[mlN]!
 * @param vctVal Array of vector elements' values. Memory will be allocated
 * during function execution via new T[mlN]!
 * @param vctNnz After function's return: Number of elements in measurement
 * vector.
 * @param ifn Name of file to read from. */
template<typename T>
void readMeasVct_HDF5(int * & vctId, T * & vctVal, int & vctNnz,
      std::string const & ifn) {
  H5Reader reader(ifn);
  int fN = reader.sizeOfFile();
  T * fMem = new T[fN];
  reader.read(fMem);
  vctNnz = findNnz(fMem, fN);
  vctId  = new int[vctNnz];
  vctVal = new T[  vctNnz];
  makeSparseVct(vctId, vctVal, fMem, fN);
  delete[] fMem;
}

/** @brief Read a measurement vector from file.
 * @tparam T Type of elements.
 * @param vctId std::vector of vector elements' indices.
 * @param vctVal std::vector of vector elements' values.
 * @param vctNnz After function's return: Number of elements in measurement
 * vector.
 * @param ifn Name of file to read from. */
template<typename T>
void readMeasVct_HDF5(std::vector<int> & vctId, std::vector<T> & vctVal,
      int & vctNnz, std::string const & ifn) {
  H5Reader reader(ifn);
  int fN = reader.sizeOfFile();
  std::vector<T> fMem(fN, 0.);
  reader.read(&fMem[0]);
  vctNnz    = findNnz(&fMem[0], fN);
  vctId.resize(vctNnz);
  vctVal.resize(vctNnz);
  makeSparseVct(&vctId[0], &vctVal[0], &fMem[0], fN);
}

/** @brief Read a measurement list from file.
 * @tparam T Type of elements in file.
 * @param ml Measurement list. Memory for the list will be allocated during
 * function execution via new int[mlN]!
 * @param mlN After function's return: Number of elements in measurement
 * list.
 * @param ifn Name of file to read from. */
template<typename T>
void readMeasList_HDF5(int * & ml, int & mlN, std::string const & ifn) {
  H5Reader reader(ifn);
  int fN = reader.sizeOfFile();
  T * fMem = new T[fN];
  reader.read(fMem);
  mlN = findNnz(fMem, fN);
  ml  = new int[mlN];
  makeMeasList(ml, fMem, fN);
  delete[] fMem;
}

/** @brief Read a measurement list from file.
 * @tparam T Type of elements in file.
 * @param ml std::vector of measurement list entries.
 * @param mlN After function's return: Number of elements in measurement
 * list.
 * @param ifn Name of file to read from. */
template<typename T>
void readMeasList_HDF5(std::vector<int> & ml, int & mlN,
      std::string const & ifn) {
  H5Reader reader(ifn);
  int fN = reader.sizeOfFile();
  std::vector<T> fMem(fN, 0.);
  reader.read(&fMem[0]);
  mlN = findNnz(&fMem[0], fN);
  ml.resize(mlN);
  makeMeasList(&ml[0], &fMem[0], fN);
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

/** @brief Number of elements in chunk.
 * @param idChunk Index of chunk.
 * @param nAll Number of elements altogether.
 * @param nFullChunk Number of elements in a full chunk.
 * @return Number of elements in chunk. */
int nInChunk(int const idChunk, int const nAll, int const nFullChunk) {
  if(idChunk < ((nAll + nFullChunk - 1) / nFullChunk)) {
    if(idChunk < (nAll / nFullChunk)) {
      return nFullChunk;
    }
    return (nAll % nFullChunk);
  }
  return 0;
}

/** @brief Index of first element in chunk.
 * @param chunkId Index of chunk.
 * @param nFullChunk Mumber of elements in a full chunk. */
int chunkPtr(int const chunkId, int const nFullChunk) {
  return chunkId*nFullChunk;
}

/** @brief Number of chunks the system matrix has to be divided into in order
 * to chunk-wise fit into limited memory. 
 * @param maxNnz Maximum number of non-zeros in the system matrix.
 * @param maxNChunk Maximum number of elements in one chunk. */
int nChunks(int const maxNnz, int const maxNChunk) {
  return ((maxNnz + maxNChunk - 1) / maxNChunk);
}
#endif	/* WRAPPERS_HPP */
