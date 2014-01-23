#include "CudaTransform.hpp"

//typedef std::complex<double> TE;
//typedef cuDoubleComplex TI;
typedef double TE;
typedef double TI;

#define NX 4
#define NY 6

int main()
{
  CudaTransform<TE,TI>  trafo;
  CudaMatrix<TE,TI>     A(NX,NY);
  CudaVector<TE,TI> x(NX);
  CudaVector<TE,TI> y(NY);
  TE alpha = 1.;
  TI beta = 0.;
  
  for(int idy=0; idy<NY; idy++) {
    for(int idx=0; idx<NX; idx++) {
      A.set(idx, idy, idy);
      x.set(idx, 1.);
    }
    y.set(idy, 0.);
  }
  
  trafo.gemv(CudaTransform<TE,TI>::base_class::BLAS_OP_N, NY, NX, &alpha, &A, NY, &x, 1, &beta, &y, 1);

  for(int idy=0; idy<NY; idy++) {
    std::cout << y.get(idy) << " ";
  }
  std::cout << std::endl;

  return 0;
}
