// Test 'normalize' method of CudaTransform
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
  CudaTransform<TE, TI>         trafo;
  SAYLINE(__LINE__+1);
  CudaVector<TE, TI>            x(N);
  SAYLINE(__LINE__+1);
  TI  norm(3.);

  /* Initialize
   * ----------
   * 
   * x =  ( 1. , 2. , 3. , 4. , 5.  )
   */
  SAYLINE(__LINE__-5);
  for(int id=0; id<N; id++)
    x.set(id, id+1.);
  
  /* Normalize */
  SAYLINE(__LINE__-1);
  trafo.normalize(&x, &norm);
  
  /* Print result - exspected :
   * --------------------------
   *
   *  ( .2, .4, .6, .8, 1. )
   */
  SAYLINES(__LINE__-5, __LINE__-1);
  for(int id=0; id<N; id++)
    std::cout << x.get(id) << " ";
  
  std::cout << std::endl;

  return 0;
}
