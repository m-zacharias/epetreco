/** @file CudaTransform.hpp */
#ifndef CUDATRANFORM_HPP
#define CUDATRANFORM_HPP

#include "Transform.hpp"
#include "CudaMatrix.hpp"
#include "CudaDeviceOnlyMatrix.hpp"
#include "CudaVector.hpp"

#include "cublas_v2.h"



template<typename TE, typename TI>
struct CudaTransformTraits
{
  public:
    
    typedef CudaMatrix<TE, TI> Matrix_t;
    typedef CudaVector<TE, TI> Vector_t;
    typedef TE                 Scalar_t;
};



template<typename TE, typename TI>
class CudaTransform : public Transform<CudaTransform<TE,TI>,
                                       CudaTransformTraits<TE,TI> >
{
  public:
    
    typedef Transform<CudaTransform<TE,TI>,
                      CudaTransformTraits<TE,TI> >         base_class;
    typedef typename base_class::Operation_t               Operation_t;
    
    typedef typename CudaTransformTraits<TE, TI>::Scalar_t Scalar_t;
    typedef typename CudaTransformTraits<TE, TI>::Matrix_t Matrix_t;
    typedef typename CudaTransformTraits<TE, TI>::Vector_t Vector_t;
    
    
    CudaTransform();
    
    ~CudaTransform();
    
    
    void gemv( Operation_t trans,
               Scalar_t * alpha,
               Matrix_t * A,
               Vector_t * x,
               Scalar_t * beta,
               Vector_t * y );

    void gemv( Operation_t trans,
               Scalar_t * alpha,
               CudaDeviceOnlyMatrix<TE, TI> * A,
               Vector_t * x,
               Scalar_t * beta,
               Vector_t * y );

    void divides( Vector_t * x,
                  Vector_t * y,
                  Vector_t * r ); 
    
    void corrects( Vector_t * x,
                   Vector_t * c,
                   Vector_t * s,
                   Vector_t * xx );

    void normalize( Vector_t * x,
                    Scalar_t * norm );
  

  private:
    
    cublasHandle_t _cublasHandle;
    cublasStatus_t _cublasStatus;
};
#include "CudaTransform.tpp"

#endif  // #define CUDATRANFORM_HPP
