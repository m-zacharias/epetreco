#include "CudaTransform.hpp"
#include "FileTalk.hpp"

//typedef std::complex<double> TE;
//typedef cuDoubleComplex TI;
typedef double TE;
typedef double TI;

#define NCOLS 4
#define NROWS 6

int main()
{
  SAYLINES(__LINE__+1, __LINE__+2);
  /* Create objects */
  CudaTransform<TE,TI>  trafo;
  SAYLINE(__LINE__+1);
  CudaMatrix<TE,TI>     A(NROWS,NCOLS);
  SAYLINE(__LINE__+1);
  CudaVector<TE,TI>     x(NCOLS);
  SAYLINE(__LINE__+1);
  CudaVector<TE,TI>     y(NROWS);
  SAYLINE(__LINE__+1);
  TE alpha = 1.;
  SAYLINES(__LINE__+1, __LINE__+2);
  TI beta = 0.;
  
  SAYLINE(__LINE__+1);
  /* Initialization
   * --------------
   *
   * x = ( 0 0 0 0 )
   *
   * y = ( 1 1 1 1 1 1 )
   *
   * A = / 0 0 0 0 \
   *     | 1 1 1 1 |
   *     | 2 2 2 2 |
   *     | 3 3 3 3 |
   *     | 4 4 4 4 |
   *     \ 5 5 5 5 /
   *
   */
  for(int rowId=0; rowId<NROWS; rowId++) {
    for(int colId=0; colId<NCOLS; colId++) {
      A.set(rowId, colId, static_cast<TE>(rowId));
      x.set(colId, 0.);
    }
    y.set(rowId, 1.);
  }
  
  SAYLINE(__LINE__+1);
  /* Matrix vector multiplication
   * ----------------------------
   * 
   *  / 0 0 0 0 \^T * ( 1 1 1 1 1 1 )
   *  | 1 1 1 1 |
   *  | 2 2 2 2 |
   *  | 3 3 3 3 |
   *  | 4 4 4 4 |
   *  \ 5 5 5 5 /
   * 
   */
  trafo.gemv(BLAS_OP_T, &alpha, &A, &y, &beta, &x);
  
  SAYLINES(__LINE__+1, __LINE__+11);
  /* Print result - exspected:
   * -------------------------
   *
   *  / 15 \
   *  | 15 |
   *  | 15 |
   *  \ 15 /
   *
   */
  for(int colId=0; colId<NCOLS; colId++) {
    std::cout << x.get(colId) << std::endl;
  }

  return 0;
}
