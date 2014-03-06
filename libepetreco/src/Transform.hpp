#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

enum BlasOperation
{
  BLAS_OP_N,
  BLAS_OP_T,
  BLAS_OP_C,
};

template<typename ConcreteTransform, typename ConcreteTransformTraits>
class Transform
{
  public:
    
    typedef typename ConcreteTransformTraits::Scalar_t Scalar_t;
    typedef typename ConcreteTransformTraits::Matrix_t Matrix_t;
    typedef typename ConcreteTransformTraits::Vector_t Vector_t;

    typedef BlasOperation                              Operation_t;

    /**
     * @brief BLAS gemv - Matrix-Vector product with a general matrix, i.e.:
     *  y = alpha*A  *x + beta*y or
     *  y = alpha*A^T*x + beta*y or
     *  y = alpha*A^H*x + beta*y
     *
     * @param trans Type of operation:  Use matrix *A in:
     *  BLAS_OP_N : non-transposed form
     *  BLAS_OP_T : transposed form
     *  BLAS_OP_C : transposed and elementwise conjugated form
     * @param M Number of rows (?)
     * @param N Number of columns (?)
     * @param alpha Pointer to the matrix scalar factor
     * @param A Pointer to the matrix
     * @param ldA Leading dimension of the matrix (which means?!)
     * @param x Pointer to the factor vector
     * @param incx Increment/spacing of the factor vector in memory
     * @param beta Pointer to the summand/output vector scalar factor
     * @param y Pointer to the summand/output vector
     * @param incy Increment/spacing of the summand/output vector in memory
     */
    void gemv( Operation_t trans, int M, int N,
               Scalar_t * alpha, Matrix_t * A, int ldA,
               Vector_t * x, int incx,
               Scalar_t * beta, Vector_t * y, int incy )
    {
      static_cast<ConcreteTransform *>(this)->\
      gemv( trans, M, N, alpha, A, ldA, x, incx, beta, y, incy );
    }
};

#endif  // #define TRANSFORM_HPP
