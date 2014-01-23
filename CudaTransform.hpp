#ifndef CUDATRANFORM_HPP
#define CUDATRANFORM_HPP

#include "Transform.hpp"
#include "CudaMatrix.hpp"
#include "CudaVector.hpp"

#include "cublas_v2.h"

template<class TE, class TI>
class CudaTransform : public Transform<TE>
{
  public:
    
    typedef Transform<TE> base_class;
    
    typedef typename base_class::Scalar_t        Scalar_t;
    typedef typename base_class::Matrix_t        Matrix_t;
    typedef typename base_class::Vector_t        Vector_t;
    typedef typename base_class::blasOperation_t blasOperation_t;

    CudaTransform();
    
    ~CudaTransform();
    
    
    void gemv( blasOperation_t trans,
      int M, int N,
      Scalar_t * alpha, Matrix_t * A, int ldA,
      Vector_t * x, int incx,
      Scalar_t * beta, Vector_t * y, int incy );
    
    
  private:
    
    cublasHandle_t _cublasHandle;
    cublasStatus_t _cublasStatus;
};
#include "CudaTransform.tpp"

#endif  // #define CUDATRANFORM_HPP
