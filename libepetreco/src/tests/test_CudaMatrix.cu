/** @file test_CudaMatrix.cu */
#include "CudaMatrix.hpp"
#include "CudaTransform.hpp"
#include <cassert>

#define NCOLS 1000
#define NROWS 100000

int main()
{
  typedef float val_t;

  CudaMatrix<val_t, val_t> A(NCOLS, NROWS);
  
  for(int colId=0; colId<NCOLS; colId++)
    for(int rowId=0; rowId<NROWS; rowId++)
      A.set(colId, rowId, 0.);

  for(int colId=0; colId<NCOLS; colId++)
  {
    for(int rowId=0; rowId<NROWS; rowId++)
    {
      val_t elem = A.get(colId, rowId);
      assert(!isnan(elem));
      assert(!isinf(elem));
    }
  }

  return 0;
}
