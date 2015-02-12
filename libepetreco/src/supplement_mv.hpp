/** 
 * @file supplement_mv.hpp
 */
 /* Author: malte
 *
 * Created on 10. Februar 2015, 13:20
 */

#ifndef SUPPLEMENT_MV_HPP
#define	SUPPLEMENT_MV_HPP

/**
 * @brief Matrix vector multiplication y = alpha * a * x + beta * y
 * @tparam T Type of elements.
 * @param alpha Scalar factor for matrix.
 * @param m Number of rows.
 * @param n Number of columns.
 * @param a Matrix in row major format.
 * @param x Dense vector of length n to multiply matrix with.
 * @param beta Scalar factor for y.
 * @param y Dense result vector of length m.
 */
template<typename T>
void mv(
      T const * const alpha, int const m, int const n, T const * const a,
      T const * const x,
      T const * const beta, T * const y) {
  for(int rowId=0; rowId<m; rowId++) {
    y[rowId] *= *beta;
    for(int colId=0; colId<n; colId++) {
      y[rowId] += *alpha * a[(rowId*n)+colId] * x[colId];
    }
  }
}



/**
 * @brief Transpose matrix vector multiplication y = alpha * a * x + beta * y
 * @tparam T Type of elements.
 * @param alpha Scalar factor for matrix.
 * @param m Number of rows.
 * @param n Number of columns.
 * @param a Matrix in row major format.
 * @param x Dense vector of length m to multiply transposed matrix with.
 * @param beta Scalar factor for y.
 * @param y Dense result vector of length n.
 */
template<typename T>
void mv_transposed(
      T const * const alpha, int const m, int const n, T const * const a,
      T const * const x,
      T const * const beta, T * const y) {
  for(int colId=0; colId<n; colId++) {
    y[colId] *= *beta;
  }
  for(int rowId=0; rowId<m; rowId++) {
    for(int colId=0; colId<n; colId++) {
      y[colId] += *alpha * a[(rowId*n)+colId] * x[rowId];
    }
  }
}



/**
 * @brief Matrix vector multiplication with a CSR matrix.
 * @tparam T Type of elements.
 * @param alpha Scalar factor for matrix.
 * @param m Number of rows in the matrix.
 * @param n Number of columns in the matrix.
 * @param nnz Number of non-zeros in the matrix.
 * @param rowPtr Array of indices in [0 .. nnz] and length m+1. For i in
 * [0 .. m-1], rowPtr[i] is the linear index of the next element starting from
 * the start of line i. rowPtr[m] is nnz.
 * @param colId Array of matrix elements' column indices in [0 .. n-1] of
 * length nnz.
 * @param val Array of matrix elements' values of length nnz.
 * @param x Dense vector of length n to multiply the matrix with.
 * @param beta Scalar factor for y.
 * @param y Dense result vector of length m.
 */
template<typename T>
void csrMv(
      T const * const alpha,
      int const m, int const n, int const nnz, int const * const rowPtr, int const * const colId, T const * const val,
      T const * const x,
      T const * const beta, T * const y) {
  // Multiply y
  for(int i=0; i<m; i++) {
    y[i] *= *beta;
  }
  
  // Go through matrix elements
  int rowId = 0;
  for(int id=0; id<nnz; id++) {
    // Increment rowId if necessary
    while(id>=rowPtr[rowId+1]) {
      rowId++;
    }
    // Stop calculation if necessary
    if(rowId>=m) {
      break;
    }
    // Add
    y[rowId] += *alpha * val[id] * x[colId[id]];
  }
}



/**
 * @brief Transpose matrix vector multiplication with a CSR matrix.
 * @tparam T Type of elements.
 * @param alpha Scalar factor for matrix.
 * @param m Number of rows in the matrix.
 * @param n Number of columns in the matrix.
 * @param nnz Number of non-zeros in the matrix.
 * @param rowPtr Array of indices in [0 .. nnz] and length m+1. For i in
 * [0 .. m-1], rowPtr[i] is the linear index of the next element starting from
 * the start of line i. rowPtr[m] is nnz.
 * @param colId Array of matrix elements' column indices in [0 .. n-1] of
 * length nnz.
 * @param val Array of matrix elements' values of length nnz.
 * @param x Dense vector of length m to multiply the transposed matrix with.
 * @param beta Scalar factor for y.
 * @param y Dense result vector of length n.
 */
template<typename T>
void csrMv_transposed(
      T const * const alpha,
      int const m, int const n, int const nnz, int const * const rowPtr, int const * const colId, T const * const val,
      T const * const x,
      T const * const beta, T * const y) {
  // Multiply y
  for(int i=0; i<m; i++) {
    y[i] *= *beta;
  }
  
  // Go through matrix elements
  int rowId = 0;
  for(int id=0; id<nnz; id++) {
    // Increment rowId if necessary
    while(id>=rowPtr[rowId+1]) {
      rowId++;
    }
    // Stop calculation if necessary
    if(rowId>=m) {
      break;
    }
    // Add
    y[colId[id]] += *alpha * val[id] * x[rowId];
  }
}

#endif	/* SUPPLEMENT_MV_HPP */

