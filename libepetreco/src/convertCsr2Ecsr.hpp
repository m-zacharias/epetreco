/** 
 * @file convertCsr2Ecsr.hpp
 */
/* Author: malte
 *
 * Created on 6. Februar 2015, 15:34
 */

#ifndef CONVERTCSR2ECSR_HPP
#define	CONVERTCSR2ECSR_HPP

/**
 * @brief Convert a matrix of CSR type to ECSR.
 * @param ecsrRowPtr
 * @param yRowId Array of matrix row ids that refer to rows that (possibly)
 * contain non-zero elements. This defines the effective column vector space.
 * Has length lengthYRowId, which is the dimension of the effective column
 * vector space.
 * @param lengthYRowId Dimension of effective column vector space.
 * @param csrRowPtr Array that is part of the definition of the matrix in CSR
 * format. Has length lengthCsrRowPtr which is the dimension of the full column
 * vector space + 1.
 * @param lengthCsrRowPtr Dimension of full column vector space + 1.
 */
void convertCsr2Ecsr(
      int * const ecsrRowPtr,
      int const * const yRowId,
      int const lengthYRowId,
      int const * const csrRowPtr,
      int const lengthCsrRowPtr) {
  for(int i=0; i<lengthYRowId; i++) {
    ecsrRowPtr[i] = csrRowPtr[yRowId[i]];
  }
  ecsrRowPtr[lengthYRowId] = csrRowPtr[lengthCsrRowPtr-1];
}

#endif	/* CONVERTCSR2ECSR_HPP */

