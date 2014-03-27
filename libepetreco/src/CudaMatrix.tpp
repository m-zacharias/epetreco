#include "CudaMatrix.hpp"

#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <complex>

// Matrix operations with cublas often need the 'leading dimension'. Because
// cublas complies with FORTRAN style column major storage, this is the
// dimension of one column - which is the number of rows!!!

template<class TE, class TI>
void CudaMatrix<TE,TI>::update_devi_data()
{
  cublasStatus_t status = cublasSetMatrix(_nRows, _nCols, sizeof(internal_elem_t),
                                          _raw_host, _nRows, _raw_devi, _nRows);
  if(status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Data upload failed." << std::endl;
    exit( EXIT_FAILURE );
  }
  _host_data_changed = false;
}

template<class TE, class TI>
void CudaMatrix<TE,TI>::update_host_data()
{
  cublasStatus_t status =
        cublasGetMatrix(_nRows, _nCols, sizeof(internal_elem_t),
                        _raw_devi, _nRows, _raw_host, _nRows );
  if(status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Data download failed." << std::endl;
    exit( EXIT_FAILURE );
  }
  _devi_data_changed = false;
}

template<class TE, class TI>    
CudaMatrix<TE,TI>::CudaMatrix( int nRows, int nCols )
: _nCols(nCols), _nRows(nRows)
{
  _raw_host = new internal_elem_t[nRows*nCols];
  cudaError_t status =
        cudaMalloc( (void**)&_raw_devi, nCols*nRows*sizeof(internal_elem_t) );
  
  if( status != cudaSuccess ) {
    std::cerr << "Device memory allocation failed: "
              << cudaGetErrorString(status) << std::endl;
    exit( EXIT_FAILURE );
  }

  _devi_data_changed = true;
  _host_data_changed = false;
}

template<class TE, class TI>
CudaMatrix<TE,TI>::~CudaMatrix()
{
  delete[] _raw_host;
  cudaFree( _raw_devi );
}

template<class TE, class TI>
int CudaMatrix<TE,TI>::getNCols()
{
  return _nCols;
}

template<class TE, class TI>
int CudaMatrix<TE,TI>::getNRows()
{
  return _nRows;
}

template<class TE, class TI>
void * CudaMatrix<TE,TI>::data()
{
  if( _host_data_changed )
    update_devi_data();
  
  return _raw_devi;
}

template<class TE, class TI>
CudaMatrix<TE,TI>::elem_t CudaMatrix<TE,TI>::get( int rowId, int colId )
{
  if( _devi_data_changed )
    update_host_data();
  
  return convert2external(_raw_host[colId * _nRows + rowId]);
}

template<class TE, class TI>
void CudaMatrix<TE,TI>::set( int rowId, int colId, elem_t val )
{
  if( _devi_data_changed )
    update_host_data();

  _raw_host[colId * _nRows + rowId] = convert2internal( val );
  _host_data_changed = true;
}

template<class TE, class TI>
CudaMatrix<TE,TI> * CudaMatrix<TE,TI>::clone()
{
  if( _host_data_changed )
    update_devi_data();
  
  CudaMatrix<TE,TI> * clone = new CudaMatrix<TE,TI>( _nCols, _nRows );
  cudaMemcpy( clone->data(), _raw_devi, _nCols*_nRows*sizeof(internal_elem_t),
              cudaMemcpyDeviceToDevice );
  
  return clone;
}

template<class TE, class TI>
void CudaMatrix<TE,TI>::set_devi_data_changed()
{
  _devi_data_changed = true;
}
