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
                          ( Operation_t trans,
                            Scalar_t * alpha,
                            Matrix_t * A,
                            Vector_t * x,
                            Scalar_t * beta,
                            Vector_t * y )
{
  TI ialpha = convert2internal(*alpha);
  TI ibeta = convert2internal(*beta);
  
  /* Explanation of cublas_gemv<>() :
   * 
   * @brief BLAS gemv - Matrix-Vector product with a general matrix, i.e.:
   *  y = alpha*A  *x + beta*y or
   *  y = alpha*A^T*x + beta*y or
   *  y = alpha*A^H*x + beta*y
   *
   * @param trans Type of operation:  Use matrix *A in:
   *  BLAS_OP_N : non-transposed form
   *  BLAS_OP_T : transposed form
   *  BLAS_OP_C : transposed and elementwise conjugated form
   * @param nRows Number of rows
   * @param nCols Number of columns
   * @param alpha Pointer to the matrix scalar factor
   * @param A Pointer to the matrix
   * @param ldA Leading dimension of the matrix (which means?!)
   * @param x Pointer to the factor vector
   * @param incx Increment/spacing of the factor vector in memory
   * @param beta Pointer to the summand/output vector scalar factor
   * @param y Pointer to the summand/output vector
   * @param incy Increment/spacing of the summand/output vector in memory
   */
  if     (trans==BLAS_OP_N) {
    _cublasStatus = cublas_gemv<TI>(_cublasHandle,
                                    CUBLAS_OP_N,
                                    A->getNRows(), A->getNCols(),
                                    &ialpha,
                                    static_cast<TI*>(A->data()), A->getNRows(),
                                    static_cast<TI*>(x->data()), 1,
                                    &ibeta,
                                    static_cast<TI*>(y->data()), 1 );
  }
  else if(trans==BLAS_OP_T) {
    _cublasStatus = cublas_gemv<TI>(_cublasHandle,
                                    CUBLAS_OP_T,
                                    A->getNRows(), A->getNCols(),
                                    &ialpha,
                                    static_cast<TI*>(A->data()), A->getNRows(),
                                    static_cast<TI*>(x->data()), 1,
                                    &ibeta,
                                    static_cast<TI*>(y->data()), 1 );
  }
  else if(trans==BLAS_OP_C) {
    _cublasStatus = cublas_gemv<TI>(_cublasHandle,
                                    CUBLAS_OP_C,
                                    A->getNRows(), A->getNCols(),
                                    &ialpha,
                                    static_cast<TI*>(A->data()), A->getNRows(),
                                    static_cast<TI*>(x->data()), 1,
                                    &ibeta,
                                    static_cast<TI*>(y->data()), 1 );
  }
  //static_cast<CudaVector<TE,TI> *>(y)->set_devi_data_changed();
  y->set_devi_data_changed();

  if(_cublasStatus != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cublas_gemv(...)  failed" << std::endl;
    // cleanup
    exit(EXIT_FAILURE);
  }
}

#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
template<typename TE, typename TI>
void CudaTransform<TE,TI>::divides( Vector_t * x, Vector_t * y, Vector_t * r )
{
  int size = x->getN();
  thrust::device_ptr<TI> xDeviPtr( static_cast<TI *>(x->data()) );
  thrust::device_ptr<TI> yDeviPtr( static_cast<TI *>(y->data()) );
  thrust::device_ptr<TI> rDeviPtr( static_cast<TI *>(r->data()) );
  
  thrust::transform( xDeviPtr, xDeviPtr+size, yDeviPtr, rDeviPtr,
                     thrust::divides<TI>() );
  
  if(cudaGetLastError()!=cudaSuccess)
  {
    std::cerr << "divides failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  r->set_devi_data_changed();
}

struct correctsFunctor
{
  template<typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<3>(t) =   thrust::get<0>(t) * thrust::get<1>(t)
                        / thrust::get<2>(t);
  }
};
template<typename TE, typename TI>
void CudaTransform<TE,TI>::corrects( Vector_t * x, Vector_t * c, Vector_t * s,
                                 Vector_t * xx )
{
  int size = x->getN();
  thrust::device_ptr<TI> xDeviPtr(  static_cast<TI *>(x-> data()) );
  thrust::device_ptr<TI> cDeviPtr(  static_cast<TI *>(c-> data()) );
  thrust::device_ptr<TI> sDeviPtr(  static_cast<TI *>(s-> data()) );
  thrust::device_ptr<TI> xxDeviPtr( static_cast<TI *>(xx->data()) );
  
  thrust::for_each(
        thrust::make_zip_iterator(thrust::make_tuple(xDeviPtr,
                                                     cDeviPtr,
                                                     sDeviPtr,
                                                     xxDeviPtr)),
        thrust::make_zip_iterator(thrust::make_tuple(xDeviPtr+size,
                                                     cDeviPtr+size,
                                                     sDeviPtr+size,
                                                     xxDeviPtr+size)),
        correctsFunctor());

  if(cudaGetLastError()!=cudaSuccess)
  {
    std::cerr << "corrects failed" << std::endl;
    exit(EXIT_FAILURE);
  }

  xx->set_devi_data_changed();
}

//template<typename TE, typename TI>
// CudaTransform<TE,TI>::
//{
//}
