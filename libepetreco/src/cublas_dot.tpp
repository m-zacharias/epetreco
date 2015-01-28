/** @file cublas_dot.tpp */
#include "cublas_dot.hpp"

// S (float)
template<>
cublasStatus_t cublas_dot<float>(
      cublasHandle_t handle,
      int N,
      float const * x, int incx,
      float const * y, int incy,
      float * result )
{
  return cublasSdot(handle, N, x, incx, y, incy, result);
}

// D (double)
template<>
cublasStatus_t cublas_dot<double>(
      cublasHandle_t handle,
      int N,
      double const * x, int incx,
      double const * y, int incy,
      double * result )
{
  return cublasDdot(handle, N, x, incx, y, incy, result);
}

// C (complex<float>)
template<>
cublasStatus_t cublas_dot<cuComplex>(
      cublasHandle_t handle,
      int N,
      cuComplex const * x, int incx,
      cuComplex const * y, int incy,
      cuComplex * result )
{
  return cublasCdotu(handle, N, x, incx, y, incy, result);
}

// Z (complex<double>)
template<>
cublasStatus_t cublas_dot<cuDoubleComplex>(
      cublasHandle_t handle,
      int N,
      cuDoubleComplex const * x, int incx,
      cuDoubleComplex const * y, int incy,
      cuDoubleComplex * result )
{
  return cublasZdotu(handle, N, x, incx, y, incy, result);
}
