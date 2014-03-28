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
     *  BLAS_OP_N : non-transposed form                         (A)
     *  BLAS_OP_T : transposed form                             (A^T)
     *  BLAS_OP_C : transposed and elementwise conjugated form  (A^H)
     * @param alpha Pointer to the matrix scalar factor
     * @param A Pointer to the matrix
     * @param x Pointer to the factor vector
     * @param beta Pointer to the summand/output vector scalar factor
     * @param y Pointer to the summand/output vector
     */
    void gemv( Operation_t trans,
               Scalar_t * alpha,
               Matrix_t * A,
               Vector_t * x,
               Scalar_t * beta,
               Vector_t * y )
    {
      static_cast<ConcreteTransform *>(this)->\
      gemv( trans, alpha, A, x, beta, y );
    }
    
    /**
     * @brief Elementwise divide Vector by Vector.
     * 
     * @param x Pointer to divident vector
     * @param y Pointer to divisor vector
     * @param r Pointer to result vector
     */
    void divides( Vector_t * x,
                  Vector_t * y,
                  Vector_t * r )
    {
      static_cast<ConcreteTransform *>(this)->\
      divides( x, y, r );
    }
    
    /**
     * @brief Elementwise calculate new intensities, not normalized.
     * 
     * @param x Pointer to old intensities vector
     * @param c Pointer to correction vector
     * @param s Pointer to sensitivity vector
     * @param xx Pointer to result vector
     */
    void corrects( Vector_t * x,
                   Vector_t * c,
                   Vector_t * s,
                   Vector_t * xx )
    {
      static_cast<ConcreteTransform *>(this)->\
      corrects( x, c, s, xx );
    }
    
    /**
     * @brief Normalize a vector
     */
    void normalize( Vector_t * x,
                    Scalar_t * norm )
    {
      static_cast<ConcreteTransform *>(this)->\
      normalize( x, norm );
    }
};

#endif  // #define TRANSFORM_HPP
