#ifndef CUBLAS_GEMV_HPP
#define CUBLAS_GEMV_HPP

#include <cublas_v2.h>

template<class T>
cublasStatus_t cublas_gemv( cublasHandle_t handle, cublasOperation_t trans,
  int M, int N, T const * alpha, T const * A, int ldA, T const * x, int incx,
  T const * beta, T * y, int incy );
#include "cublas_gemv.tpp"

#endif  // #define CUBLAS_GEMV_HPP
