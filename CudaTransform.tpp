#include "CudaTransform.hpp"

#include "cublas_gemv.hpp"
#include "conversion.hpp"
#include <iostream>
#include <stdlib.h>

template<typename TE, typename TI>
CudaTransform<TE,TI>::CudaTransform()
{
  _cublasStatus = cublasCreate( &_cublasHandle );
  
  if(_cublasStatus != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Cublas initialization failed" << std::endl;
    // cleanup
    exit(EXIT_FAILURE);
  }
}

template<typename TE, typename TI>
CudaTransform<TE,TI>::~CudaTransform()
{
  cublasDestroy(_cublasHandle);
}

template<typename TE, typename TI>
void CudaTransform<TE,TI>::gemv
                          ( Operation_t trans, int M, int N,
                            Scalar_t * alpha, Matrix_t * A, int ldA,
                            Vector_t * x, int incx,
                            Scalar_t * beta, Vector_t * y, int incy )
{
  TI ialpha = convert2internal(*alpha);
  TI ibeta = convert2internal(*beta);
  
  if     (trans==BLAS_OP_N) {
    _cublasStatus = cublas_gemv<TI>(_cublasHandle, CUBLAS_OP_N, M, N,
                                    &ialpha, static_cast<TI*>(A->data()), ldA,
                                    static_cast<TI*>(x->data()), incx,
                                    &ibeta, static_cast<TI*>(y->data()), incy );
  }
  else if(trans==BLAS_OP_T) {
    _cublasStatus = cublas_gemv<TI>(_cublasHandle, CUBLAS_OP_T, M, N,
                                    &ialpha, static_cast<TI*>(A->data()), ldA,
                                    static_cast<TI*>(x->data()), incx,
                                    &ibeta, static_cast<TI*>(y->data()), incy );
  }
  else if(trans==BLAS_OP_C) {
    _cublasStatus = cublas_gemv<TI>(_cublasHandle, CUBLAS_OP_C, M, N,
                                    &ialpha, static_cast<TI*>(A->data()), ldA,
                                    static_cast<TI*>(x->data()), incx,
                                    &ibeta, static_cast<TI*>(y->data()), incy );
  }
  static_cast<CudaVector<TE,TI> *>(y)->set_devi_data_changed();

  if(_cublasStatus != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublas_gemv(...)  failed" << std::endl;
    // cleanup
    exit(EXIT_FAILURE);
  }
}

//template<typename TE, typename TI>
// CudaTransform<TE,TI>::
//{
//}
