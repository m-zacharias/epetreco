#ifndef TRANSFORM_HPP
#define TRANSFORM_HPP

#include "Matrix.hpp"
#include "Vector.hpp"


template<class Scalar>
class Transform
{
  public:
    
    typedef Scalar           Scalar_t;
    typedef Matrix<Scalar_t> Matrix_t;
    typedef Vector<Scalar_t> Vector_t;
    
    /**
     * @brief Enumeration type, for specifying type of operation mode in BLAS
     *  operations that have different operation modes:
     *  BLAS_OP_N : non-transposed
     *  BLAS_OP_T : transposed
     *  BLAS_OP_C : transposed + complex conjugated
     */
    enum blasOperation_t
    {
      BLAS_OP_N,
      BLAS_OP_T,
      BLAS_OP_C
    };
    
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
    virtual void gemv( blasOperation_t trans,
      int M, int N,
      Scalar_t * alpha, Matrix<Scalar_t> * A, int ldA,
      Vector<Scalar_t> * x, int incx,
      Scalar_t * beta, Vector<Scalar_t> * y, int incy ) = 0;
};

#endif  // #define TRANSFORM_HPP
