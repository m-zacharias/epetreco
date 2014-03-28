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
   * x = ( 1 1 1 1 )
   *
   * A = / 0 0 0 0 \   y = / 0 \
   *     | 1 1 1 1 |       | 0 |
   *     | 2 2 2 2 |       | 0 |
   *     | 3 3 3 3 |       | 0 |
   *     | 4 4 4 4 |       | 0 |
   *     \ 5 5 5 5 /       \ 0 /
   *
   */
  for(int rowId=0; rowId<NROWS; rowId++) {
    for(int colId=0; colId<NCOLS; colId++) {
      A.set(rowId, colId, static_cast<TE>(rowId));
      x.set(colId, 1.);
    }
    y.set(rowId, 0.);
  }
  
  SAYLINE(__LINE__+1);
  /* Matrix vector multiplication
   * ----------------------------
   * 
   *  / 0 0 0 0 \ * ( 1 1 1 1 )
   *  | 1 1 1 1 |
   *  | 2 2 2 2 |
   *  | 3 3 3 3 |
   *  | 4 4 4 4 |
   *  \ 5 5 5 5 /
   * 
   */
  trafo.gemv(BLAS_OP_N, &alpha, &A, &x, &beta, &y);
  
  SAYLINES(__LINE__+1, __LINE__+11);
  /* Print result - exspected:
   * -------------------------
   *
   *  /  0 \
   *  |  4 |
   *  |  8 |
   *  | 12 |
   *  | 16 |
   *  \ 20 /
   *
   */
  for(int rowId=0; rowId<NROWS; rowId++) {
    std::cout << y.get(rowId) << std::endl;
  }

  return 0;
}
