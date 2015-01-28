/** @file cublas_dot.hpp */
#ifndef CUBLAS_DOT_HPP
#define CUBLAS_DOT_HPP

#include <cublas_v2.h>

template<typename T>
cublasStatus_t cublas_dot( cublasHandle_t handle,
                           int N,
                           T const * x, int incx,
                           T const * y, int incy,
                           T * result );
#include "cublas_dot.tpp"

#endif  // #ifndef CUBLAS_DOT_HPP
