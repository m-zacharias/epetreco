/** @file test_CudaTransform4.cu */
// Test elementwise 'corrects' method of Cudatransform
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
  CudaVector<TE, TI>    c(N);
  SAYLINE(__LINE__+1);
  CudaVector<TE, TI>    s(N);
  SAYLINE(__LINE__+1);
  CudaVector<TE, TI>    xx(N);

  /* Initialize
   * ----------
   * 
   * x =  ( 3. , 3. , 3. , 3. , 3.  )
   * c =  (  .4,  .4,  .5,  .6,  .6 )
   * s =  (  .5,  .5,  .5,  .5,  .5 )
   * xx = ( 0. , 0. , 0. , 0. , 0.  )
   */
  SAYLINE(__LINE__-8);
  for(int id=0; id<N; id++)
  {
    x.set( id, 3.);
    
    if     (id<2) c.set( id,  .8);
    else if(id<3) c.set( id, 1. );
    else if(id<5) c.set( id, 1.2);
    
    s.set( id, 1.);
    xx.set(id, 0.);
  }

  /* Elementwise correct */
  SAYLINE(__LINE__-1);
  trafo.corrects(&x, &c, &s, &xx);

  /* Print result - exspected :
   * --------------------------
   *
   *  ( 2.4, 2.4, 3., 3.6, 3.6 )
   */
  SAYLINES(__LINE__-5, __LINE__-1);
  for(int id=0; id<N; id++)
    std::cout << xx.get(id) << std::endl;

  return 0;
}
