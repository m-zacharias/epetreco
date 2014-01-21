#ifndef CUDAMATRIX_HPP
#define CUDAMATRIX_HPP

#include "Matrix.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>

template<class T1, class T2>
class CudaMatrix : public Matrix<T1>
{
  private:

    void update_devi_data( void );
    
    void update_host_data( void );
    
//    cuDoubleComplex * raw_host_;
    T2 * raw_host_;
    
//    cuDoubleComplex * raw_devi_;
    T2 * raw_devi_;
    
    bool devi_data_changed_;
    
    bool host_data_changed_;
    
    int nx_;
    
    int ny_;
  
    
  public:

    CudaMatrix( int nx, int ny );
    
    ~CudaMatrix();
    
    int get_nx( void );
    
    int get_ny( void );
    
    void * data( void );
    
//    std::complex<double> get( int idx, int idy );
    T1 get( int idx, int idy );
    
//    void set( int idx, int idy, std::complex<double> val );
    void set( int idx, int idy, T1 val );
    
    Matrix * clone( void );
    
    void set_devi_data_changed( void );
};
#include "CudaMatrix.tpp"

#endif  // #define CUDAMATRIX_HPP
