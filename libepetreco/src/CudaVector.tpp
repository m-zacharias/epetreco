#include "CudaVector.hpp"

#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <complex>

#include "conversion.hpp"

template<class TE, class TI>
void CudaVector<TE,TI>::update_devi_data()
{
  cublasStatus_t cublasStatus = cublasSetVector(_n, sizeof(internal_elem_t), _raw_host, 1,
                                                _raw_devi, 1);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Data upload failed." << std::endl;
    exit(EXIT_FAILURE);
  }
  _host_data_changed = false;
}

template<class TE, class TI>
void CudaVector<TE,TI>::update_host_data()
{
  cublasStatus_t cublasStatus = cublasGetVector(_n, sizeof(internal_elem_t), _raw_devi, 1,
                                                _raw_host, 1);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Data download failed." << std::endl;
    exit(EXIT_FAILURE);
  }
  _devi_data_changed = false;
}

template<class TE, class TI>
CudaVector<TE,TI>::CudaVector( int n )
: _n(n)
{
  _raw_host = new internal_elem_t[n];

  cudaError_t cudaError = cudaMalloc((void**)&_raw_devi, n*sizeof(internal_elem_t));
  if(cudaError != cudaSuccess) {
    std::cerr << "Device memory allocation failed: "
              << cudaGetErrorString(cudaError) <<std::endl;
    exit(EXIT_FAILURE);
  }
  _devi_data_changed = true;
  _host_data_changed = false;
}

template<class TE, class TI>
CudaVector<TE,TI>::~CudaVector()
{
  delete[] _raw_host;
  cudaFree(_raw_devi);
}

template<class TE, class TI>
int CudaVector<TE,TI>::getN()
{
  return _n;
}

template<class TE, class TI>
void * CudaVector<TE,TI>::data()
{
  if(_host_data_changed)
    update_devi_data();

  return _raw_devi;
}

template<class TE, class TI>
TE CudaVector<TE,TI>::get( int id )
{
  if(_devi_data_changed)
    update_host_data();
  
  return TE( _raw_host[id] );
}

template<class TE, class TI>
void CudaVector<TE,TI>::set( int id, TE val )
{
  if(_devi_data_changed)
    update_host_data();
  
  _raw_host[id] = convert2internal(val);
  _host_data_changed = true;
}

template<class TE, class TI>
CudaVector<TE,TI> * CudaVector<TE,TI>::clone()
{
  if(_host_data_changed)
    update_devi_data();

  CudaVector<TE,TI> * clone = new CudaVector<TE,TI>(_n);
  cudaMemcpy(clone->data(), _raw_devi, _n*sizeof(internal_elem_t),
             cudaMemcpyDeviceToDevice);

  return clone;
}

template<class TE, class TI>
void CudaVector<TE,TI>::set_devi_data_changed()
{
  _devi_data_changed = true;
}
