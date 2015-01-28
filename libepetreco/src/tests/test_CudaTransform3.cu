/** @file test_CudaTransform3.cu */
// Test elementwise division method of CudaTransform
#include "CudaTransform.hpp"
#include "FileTalk.hpp"
#include <iostream>

#define N 5

typedef float TE;
typedef float TI;

int main()
{
  /* Create objects */
  SAYLINE(__LINE__-1);
  SAYLINE(__LINE__+1);
  CudaTransform<TE, TI> trafo;
  SAYLINE(__LINE__+1);
  CudaVector<TE, TI>    x(N);
  SAYLINE(__LINE__+1);
  CudaVector<TE, TI>    y(N);
  SAYLINE(__LINE__+1);
  CudaVector<TE, TI>    r(N);

  /* Initialize
   * ----------
   * 
   * x = ( 7  , 14  , 21  , 28  , 35   )
   * y = ( 1  ,  2  ,  3  ,  4  ,  5   )
   * r = ( 1.5,  1.5,  1.5,  1.5,  1.5 )
   */
  SAYLINE(__LINE__-7);
  for(int id=0; id<N; id++)
  {
    x.set(id, (id+1)*7);
    y.set(id, (id+1));
    r.set(id, 1.5);
  }
  
  /* Elementwise division
   * --------------------
   *
   * r = /  7  /  1 \
   *     | 14  /  2 |
   *     | 21  /  3 |
   *     | 28  /  4 |
   *     \ 35  /  5 /
   */
  SAYLINE(__LINE__-9);
  trafo.divides(&x, &y, &r);
  
  /* Print result - exspected :
   * --------------------------
   *
   *  / 7 \
   *  | 7 |
   *  | 7 |
   *  | 7 |
   *  \ 7 /
   */
  SAYLINES(__LINE__-9, __LINE__-1);
  for(int id=0; id<N; id++)
    std::cout << r.get(id) << std::endl;

  return 0;
}
