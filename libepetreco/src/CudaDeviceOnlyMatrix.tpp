/** @file CudaDeviceOnlyMatrix.tpp */
#include "CudaDeviceOnlyMatrix.hpp"

#include <iostream>
#include <cstdlib>

template<typename TE, typename TI>
CudaDeviceOnlyMatrix<TE, TI>::CudaDeviceOnlyMatrix( int nRows, int nCols )
: _nRows(nRows), _nCols(nCols)
{
  cudaError_t status =
        cudaMalloc((void**)&_raw_devi, nCols*nRows*sizeof(internal_elem_t));

  if(status != cudaSuccess)
  {
    std::cerr << "Device memory allocation failed: "
              << cudaGetErrorString(status) << std::endl;
    exit(EXIT_FAILURE);
  }
}

template<typename TE, typename TI>
CudaDeviceOnlyMatrix<TE, TI>::~CudaDeviceOnlyMatrix()
{
  cudaFree(_raw_devi);
}

template<typename TE, typename TI>
int CudaDeviceOnlyMatrix<TE, TI>::getNRows()
{
  return _nRows;
}

template<typename TE, typename TI>
int CudaDeviceOnlyMatrix<TE, TI>::getNCols()
{
  return _nCols;
}

template<typename TE, typename TI>
void * CudaDeviceOnlyMatrix<TE, TI>::data()
{
  return _raw_devi;
}

template<typename TE, typename TI>
CudaDeviceOnlyMatrix<TE, TI>::elem_t CudaDeviceOnlyMatrix<TE, TI>::get( int rowId, int colId )
{
  internal_elem_t temp;
  cudaError_t status =
        cudaMemcpy(&temp, &_raw_devi[colId * _nRows + rowId], sizeof(internal_elem_t),
                   cudaMemcpyDeviceToHost);
  status = cudaDeviceSynchronize();

  if(status != cudaSuccess)
  {
    std::cerr << "cudaMemcpy failed: "
              << cudaGetErrorString(status) << std::endl;
    exit(EXIT_FAILURE);
  }

  return convert2external(temp);
}

template<typename TE, typename TI>
void CudaDeviceOnlyMatrix<TE, TI>::set( int rowId, int colId, elem_t val )
{
  internal_elem_t temp = convert2internal(val);
  cudaError_t status =
        cudaMemcpy(&_raw_devi[colId * _nRows + rowId], &temp, sizeof(internal_elem_t),
                   cudaMemcpyHostToDevice);
  status = cudaDeviceSynchronize();
  
  if(status != cudaSuccess)
  {
    std::cerr << "cudaMemcpy failed: "
              << cudaGetErrorString(status) << std::endl;
    exit(EXIT_FAILURE);
  }
}

template<typename TE, typename TI>
CudaDeviceOnlyMatrix<TE, TI> * CudaDeviceOnlyMatrix<TE, TI>::clone()
{
  CudaDeviceOnlyMatrix<TE, TI> * clone = new CudaMatrix<TE, TI>(_nCols, _nRows);
  cudaError_t status =
        cudaMemcpy(clone->data(), _raw_devi, _nCols*_nRows*sizeof(internal_elem_t),
                   cudaMemcpyDeviceToDevice);
  if(status != cudaSuccess)
  {
    std::cerr << "cudaMemcpy failed: "
              << cudaGetErrorString(status) << std::endl;
    exit(EXIT_FAILURE);
  }
  
  return clone;
}
