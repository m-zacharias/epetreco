#include "CudaTransform.hpp"
#include "CudaDeviceOnlyMatrix.hpp"
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
  SAYLINE(__LINE__+1);
  CudaDeviceOnlyMatrix<TE,TI>     A(NROWS,NCOLS);
  
  SAYLINE(__LINE__+1);
  /* Initialization
   * --------------
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
    }
  }
  
  SAYLINES(__LINE__+1, __LINE__+3);
  /* Print matrix
   * -------------------------
   */
  for(int rowId=0; rowId<NROWS; rowId++) {
    for(int colId=0; colId<NCOLS; colId++) {
      std::cout << A.get(rowId, colId) << "  ";
    }
    std::cout << std::endl;
  }

  return 0;
}
