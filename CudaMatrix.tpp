#include "CudaMatrix.hpp"
#include <iostream>
#include <cstdlib>

template<class T1, class T2>
void CudaMatrix<T1,T2>::update_devi_data( void )
{
  cublasStatus_t status = cublasSetMatrix(ny_, nx_, sizeof(T2),
                                          raw_host_, ny_, raw_devi_, ny_);
  if(status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Data upload failed." << std::endl;
    exit( EXIT_FAILURE );
  }
  host_data_changed_ = false;
}

template<class T1, class T2>
void CudaMatrix<T1,T2>::update_host_data( void )
{
  cublasStatus_t status = cublasGetMatrix(ny_, nx_, sizeof(T2),
                                          raw_devi_, ny_, raw_host_, ny_ );
  if(status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Data download failed." << std::endl;
    exit( EXIT_FAILURE );
  }
  devi_data_changed_ = false;
}

template<class T1, class T2>    
CudaMatrix<T1,T2>::CudaMatrix( int nx, int ny )
: nx_(nx), ny_(ny)
{
  raw_host_ = new T2[nx*ny];
  cudaError_t status = cudaMalloc( (void**)&raw_devi_, nx_*ny_*sizeof(T2) );
  if( status != cudaSuccess ) {
    std::cerr << "Device memory allocation failed (code "
              << status << ")." << std::endl;
    exit( EXIT_FAILURE );
  }
  devi_data_changed_ = true;
  host_data_changed_ = false;
}

template<class T1, class T2>
CudaMatrix<T1,T2>::~CudaMatrix()
{
  delete[] raw_host_;
  cudaFree( raw_devi_ );
}

template<class T1, class T2>
int CudaMatrix<T1,T2>::get_nx( void )
{
  return nx_;
}

template<class T1, class T2>
int CudaMatrix<T1,T2>::get_ny( void )
{
  return ny_;
}

template<class T1, class T2>
void * CudaMatrix<T1,T2>::data( void )
{
  if( host_data_changed_ )
    update_devi_data();
  return raw_devi_;
}

template<class T1, class T2>
T1 CudaMatrix<T1,T2>::get( int idx, int idy )
{
  if( devi_data_changed_ )
    update_host_data();
  return T1( raw_host_[idx * ny_ + idy].x, raw_host_[idx * ny_ + idy].y );
}

template<class T1, class T2>
void CudaMatrix<T1,T2>::set( int idx, int idy, T1 val )
{
  if( devi_data_changed_ )
    update_host_data();
  T2 tmp = make_cuDoubleComplex( val.real(), val.imag() );
  raw_host_[idx * ny_ + idy] = tmp;
  host_data_changed_ = true;
}

template<class T1, class T2>
Matrix<T1> * CudaMatrix<T1,T2>::clone( void )
{
  if( host_data_changed_ )
    update_devi_data();
  CudaMatrix<T1,T2> * clone = new CudaMatrix<T1,T2>( nx_, ny_ );
  cudaMemcpy( clone->data(), raw_devi_, nx_*ny_*sizeof(T2), cudaMemcpyDeviceToDevice );
  return clone;
}

template<class T1, class T2>
void CudaMatrix<T1,T2>::set_devi_data_changed( void )
{
  devi_data_changed_ = true;
}
