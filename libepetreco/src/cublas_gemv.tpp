/** @file cublas_gemv.tpp */
#include "cublas_gemv.hpp"

// S (float)
template<>
cublasStatus_t cublas_gemv<float>( cublasHandle_t handle,
  cublasOperation_t trans, int M, int N, float const * alpha, float const * A,
  int ldA, float const * x, int incx, float const * beta, float * y, int incy )
{
  return cublasSgemv(handle, trans, M, N, alpha, A, ldA, x, incx, beta, y, incy);
}

// D (double)
template<>
cublasStatus_t cublas_gemv<double>( cublasHandle_t handle,
  cublasOperation_t trans, int M, int N, double const * alpha, double const * A,
  int ldA, double const * x, int incx, double const * beta, double * y,
  int incy )
{
  return cublasDgemv(handle, trans, M, N, alpha, A, ldA, x, incx, beta, y, incy);
}

// C (complex<float>)
template<>
cublasStatus_t cublas_gemv<cuComplex>( cublasHandle_t handle,
  cublasOperation_t trans, int M, int N, cuComplex const * alpha,
  cuComplex const * A, int ldA, cuComplex const * x, int incx,
  cuComplex const * beta, cuComplex * y, int incy )
{
  return cublasCgemv(handle, trans, M, N, alpha, A, ldA, x, incx, beta, y, incy);
}

// Z (complex<double>)
template<>
cublasStatus_t cublas_gemv<cuDoubleComplex>( cublasHandle_t handle,
  cublasOperation_t trans, int M, int N, cuDoubleComplex const * alpha,
  cuDoubleComplex const * A, int ldA, cuDoubleComplex const * x, int incx,
  cuDoubleComplex const * beta, cuDoubleComplex * y, int incy )
{
  return cublasZgemv(handle, trans, M, N, alpha, A, ldA, x, incx, beta, y, incy);
}
