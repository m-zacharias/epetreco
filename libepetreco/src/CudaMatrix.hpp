#ifndef CUDAMATRIX_HPP
#define CUDAMATRIX_HPP

#include "Matrix.hpp"



template<typename TE, typename TI>
struct CudaMatrixTraits
{
  public:
    
    typedef TE                elem_t;
    typedef TE                external_elem_t;
    typedef TI                internal_elem_t;
};



template<class TE, class TI>
class CudaMatrix : public Matrix<CudaMatrix<TE,TI>, CudaMatrixTraits<TE,TI> >
{
  public:
    
    typedef typename CudaMatrixTraits<TE, TI>::elem_t          elem_t;
    typedef typename CudaMatrixTraits<TE, TI>::external_elem_t external_elem_t;
    typedef typename CudaMatrixTraits<TE, TI>::internal_elem_t internal_elem_t;

    CudaMatrix( int nx, int ny );
    
    ~CudaMatrix();
    

    int getNRows();
    
    int getNCols();
    
    void * data();
    
    elem_t get( int rowId, int colId );
    
    void set( int rowId, int colId, elem_t val );
    
    CudaMatrix<TE,TI> * clone();
    

    void set_devi_data_changed();
    
    
  private:

    void update_devi_data();
    
    void update_host_data();
    

    internal_elem_t * _raw_host;
    
    internal_elem_t * _raw_devi;
    
    bool _devi_data_changed;
    
    bool _host_data_changed;
    
    int _nRows;
    
    int _nCols;
};
#include "CudaMatrix.tpp"

#endif  // #define CUDAMATRIX_HPP
