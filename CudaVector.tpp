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
  cublasStatus_t cublasStatus = cublasSetVector(n_, sizeof(internal_elem_t), raw_host_, 1,
                                                raw_devi_, 1);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Data upload failed." << std::endl;
    exit(EXIT_FAILURE);
  }
  host_data_changed_ = false;
}

template<class TE, class TI>
void CudaVector<TE,TI>::update_host_data()
{
  cublasStatus_t cublasStatus = cublasGetVector(n_, sizeof(internal_elem_t), raw_devi_, 1,
                                                raw_host_, 1);
  if(cublasStatus != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Data download failed." << std::endl;
    exit(EXIT_FAILURE);
  }
  devi_data_changed_ = false;
}

template<class TE, class TI>
CudaVector<TE,TI>::CudaVector( int n )
: n_(n)
{
  raw_host_ = new internal_elem_t[n];

  cudaError_t cudaError = cudaMalloc((void**)&raw_devi_, n*sizeof(internal_elem_t));
  if(cudaError != cudaSuccess) {
    std::cerr << "Device memory allocation failed: "
              << cudaGetErrorString(cudaError) <<std::endl;
    exit(EXIT_FAILURE);
  }
  devi_data_changed_ = true;
  host_data_changed_ = false;
}

template<class TE, class TI>
CudaVector<TE,TI>::~CudaVector()
{
  delete[] raw_host_;
  cudaFree(raw_devi_);
}

template<class TE, class TI>
int CudaVector<TE,TI>::get_n()
{
  return n_;
}

template<class TE, class TI>
void * CudaVector<TE,TI>::data()
{
  if(host_data_changed_)
    update_devi_data();

  return raw_devi_;
}

template<class TE, class TI>
TE CudaVector<TE,TI>::get( int id )
{
  if(devi_data_changed_)
    update_host_data();
  
  return TE( raw_host_[id] );
}

template<class TE, class TI>
void CudaVector<TE,TI>::set( int id, TE val )
{
  if(devi_data_changed_)
    update_host_data();
  
  raw_host_[id] = convert2internal(val);
  host_data_changed_ = true;
}

template<class TE, class TI>
CudaVector<TE,TI> * CudaVector<TE,TI>::clone()
{
  if(host_data_changed_)
    update_devi_data();

  CudaVector<TE,TI> * clone = new CudaVector<TE,TI>(n_);
  cudaMemcpy(clone->data(), raw_devi_, n_*sizeof(internal_elem_t), cudaMemcpyDeviceToDevice);

  return clone;
}

template<class TE, class TI>
void CudaVector<TE,TI>::set_devi_data_changed()
{
  devi_data_changed_ = true;
}
