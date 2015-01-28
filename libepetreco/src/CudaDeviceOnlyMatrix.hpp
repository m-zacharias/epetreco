/** @file CudaDeviceOnlyMatrix.hpp */
#ifndef CUDADEVICEONLYMATRIX_HPP
#define CUDADEVICEONLYMATRIX_HPP

#include "Matrix.hpp"

template<typename TE, typename TI>
struct CudaDeviceOnlyMatrixTraits
{
  typedef TE  elem_t;
  typedef TE  external_elem_t;
  typedef TI  internal_elem_t;
};

template<typename TE, typename TI>
class CudaDeviceOnlyMatrix
: public Matrix<CudaDeviceOnlyMatrix<TE, TI>, CudaDeviceOnlyMatrixTraits<TE, TI> >
//: public CudaMatrix<TE, TI>
{
  public:
    
    typedef typename CudaDeviceOnlyMatrixTraits<TE, TI>::elem_t          elem_t;
    typedef typename CudaDeviceOnlyMatrixTraits<TE, TI>::external_elem_t external_elem_t;
    typedef typename CudaDeviceOnlyMatrixTraits<TE, TI>::internal_elem_t internal_elem_t;

    CudaDeviceOnlyMatrix( int nRows, int nCols );

    ~CudaDeviceOnlyMatrix();


    int getNRows();

    int getNCols();
    
    void * data();

    elem_t get( int rowId, int colId );

    void set( int rowId, int colId, elem_t val );

    CudaDeviceOnlyMatrix<TE, TI> * clone();


  private:
    
    internal_elem_t * _raw_devi;

    int _nRows;

    int _nCols;
};
#include "CudaDeviceOnlyMatrix.tpp"

#endif  // #define CUDADEVICEONLYMATRIX_HPP
