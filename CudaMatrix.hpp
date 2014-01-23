#ifndef CUDAMATRIX_HPP
#define CUDAMATRIX_HPP

#include "Matrix.hpp"

template<class TE, class TI>
class CudaMatrix : public Matrix<TE>
{
  public:
    
    typedef TI internal_elem_t;
    

    CudaMatrix( int nx, int ny );
    
    ~CudaMatrix();
    

    int get_nx();
    
    int get_ny();
    
    void * data();
    
    TE get( int idx, int idy );
    
    void set( int idx, int idy, TE val );
    
    Matrix<TE> * clone();
    

    void set_devi_data_changed();
    
    
  private:

    void update_devi_data();
    
    void update_host_data();
    

    TI * raw_host_;
    
    TI * raw_devi_;
    
    bool devi_data_changed_;
    
    bool host_data_changed_;
    
    int nx_;
    
    int ny_;
};
#include "CudaMatrix.tpp"

#endif  // #define CUDAMATRIX_HPP
