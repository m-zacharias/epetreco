#include "CudaMatrix.hpp"

#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <complex>

template<class TE, class TI>
void CudaMatrix<TE,TI>::update_devi_data()
{
  cublasStatus_t status = cublasSetMatrix(ny_, nx_, sizeof(internal_elem_t),
                                          raw_host_, ny_, raw_devi_, ny_);
  if(status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Data upload failed." << std::endl;
    exit( EXIT_FAILURE );
  }
  host_data_changed_ = false;
}

template<class TE, class TI>
void CudaMatrix<TE,TI>::update_host_data()
{
  cublasStatus_t status = cublasGetMatrix(ny_, nx_, sizeof(internal_elem_t),
                                          raw_devi_, ny_, raw_host_, ny_ );
  if(status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Data download failed." << std::endl;
    exit( EXIT_FAILURE );
  }
  devi_data_changed_ = false;
}

template<class TE, class TI>    
CudaMatrix<TE,TI>::CudaMatrix( int nx, int ny )
: nx_(nx), ny_(ny)
{
  raw_host_ = new internal_elem_t[nx*ny];
  cudaError_t status = cudaMalloc( (void**)&raw_devi_, nx_*ny_*sizeof(internal_elem_t) );
  
  if( status != cudaSuccess ) {
    std::cerr << "Device memory allocation failed: "
              << cudaGetErrorString(status) << std::endl;
    exit( EXIT_FAILURE );
  }

  devi_data_changed_ = true;
  host_data_changed_ = false;
}

template<class TE, class TI>
CudaMatrix<TE,TI>::~CudaMatrix()
{
  delete[] raw_host_;
  cudaFree( raw_devi_ );
}

template<class TE, class TI>
int CudaMatrix<TE,TI>::get_nx()
{
  return nx_;
}

template<class TE, class TI>
int CudaMatrix<TE,TI>::get_ny()
{
  return ny_;
}

template<class TE, class TI>
void * CudaMatrix<TE,TI>::data()
{
  if( host_data_changed_ )
    update_devi_data();
  
  return raw_devi_;
}

template<class TE, class TI>
CudaMatrix<TE,TI>::elem_t CudaMatrix<TE,TI>::get( int idx, int idy )
{
  if( devi_data_changed_ )
    update_host_data();
  
  return convert2external(raw_host_[idx * ny_ + idy]);
}

template<class TE, class TI>
void CudaMatrix<TE,TI>::set( int idx, int idy, elem_t val )
{
  if( devi_data_changed_ )
    update_host_data();

  raw_host_[idx * ny_ + idy] = convert2internal( val );
  host_data_changed_ = true;
}

template<class TE, class TI>
CudaMatrix<TE,TI> * CudaMatrix<TE,TI>::clone()
{
  if( host_data_changed_ )
    update_devi_data();
  
  CudaMatrix<TE,TI> * clone = new CudaMatrix<TE,TI>( nx_, ny_ );
  cudaMemcpy( clone->data(), raw_devi_, nx_*ny_*sizeof(internal_elem_t), cudaMemcpyDeviceToDevice );
  
  return clone;
}

template<class TE, class TI>
void CudaMatrix<TE,TI>::set_devi_data_changed()
{
  devi_data_changed_ = true;
}
