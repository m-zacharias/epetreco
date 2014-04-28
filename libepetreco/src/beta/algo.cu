/* Reconstruction development example.  Uses a small and simple 2D geomtry.
 * 
 * For the chosen setup (detectors, voxel grid), a true density vector xx is
 * defined.  Using the system matrix, its "true" measurement vector y is
 * calculated.  The vector that holds the density guess is x, its elements are
 * initialized with a constant value.  This guess is then iteratively improved
 * using MLEM.
 */

#include "defines.h"
#include "CUDA_HandleError.hpp"
#include "ChordsCalc_kernel2.cu"
#include "CudaTransform.hpp"
#include <iomanip>

#define M VGRIDSIZE // number of voxels
#define N NCHANNELS // number of channels

struct Grid
{
  Grid( float const gridO0, float const gridO1, float const gridO2,
        float const gridD0, float const gridD1, float const gridD2,
        int const   gridN0, int const   gridN1, int const   gridN2 )
  {
    gridO[0]=gridO0; gridO[1]=gridO1; gridO[2]=gridO2;
    gridD[0]=gridD0; gridD[1]=gridD1; gridD[2]=gridD2;
    gridN[0]=gridN0; gridN[1]=gridN1; gridN[2]=gridN2;
  }

  float gridO[3];
  float gridD[3];
  int   gridN[3];
};

void chordsCalc(
      int const nBlocks, int const nThreads,
      float * const chords,
      float * const rays,
      Grid const * const grid )
{
  chordsCalc<<<nBlocks, nThreads>>>(chords, rays, grid->gridO, grid->gridD, grid->gridN);
}



int main()
{
  /* Create objects */
  CudaTransform<float,float>        trafo;
  CudaDeviceOnlyMatrix<float,float> SM(N, M);       // system matrix
  CudaVector<float,float>           xx(M);          // true density
  CudaVector<float,float>           x(M);           // density guess
  for(int voxelId=0; voxelId<M; voxelId++)
  {
    if(voxelId == 6) xx.set(voxelId, 1.);
    x.set(voxelId, 1.);
  }
  CudaVector<float,float>           e(N);           // simulated measurement
  CudaVector<float,float>           y(N);           // true measurement
  CudaVector<float,float>           yy(N);          // "error"
  CudaVector<float,float>           c(M);           // correction
  CudaVector<float,float>           s(M);           // sensitivity

  float * rays_devi;
  HANDLE_ERROR( cudaMalloc((void**)&rays_devi, N*NTHREADRAYS*sizeof(float)) );
  
  Grid * grid_host = new Grid(-1.5,-0.5,-2.0,1.0,1.0,1.0,GRIDNX,GRIDNY,GRIDNZ);
  Grid * grid_devi;
  HANDLE_ERROR( cudaMalloc((void**)&grid_devi, sizeof(Grid)) );
  HANDLE_ERROR( cudaMemcpy(grid_devi, grid_host, sizeof(Grid),
                           cudaMemcpyHostToDevice) );
  
  /* Calculate system matrix */
  chordsCalc(N, 1, static_cast<float*>(SM.data()), rays_devi, grid_devi);
  HANDLE_ERROR( cudaDeviceSynchronize() );

  std::cout << "System matrix:" << std::endl
            << "--------------" << std::endl;
  std::cout << std::setprecision(2);
  for(int i=0; i<N; i++)
  {
    for(int j=0; j<M; j++)
      std::cout << std::setw(7) << SM.get(i, j);
    std::cout << std::endl;
  }
  
  float one(1.);
  float zero(0.);

  /* Calculate "true" measurement */
  trafo.gemv(BLAS_OP_N,
             &one, &SM,
             &xx,
             &zero, &y);
  
  std::cout << std::endl
            << "\"True\" measurement vector:" << std::endl
            << "----------------------------" << std::endl;
  for(int channelId=0; channelId<N; channelId++)
    std::cout << y.get(channelId) << std::endl;

  /*Calculate sensitivity */
  CudaVector<float,float>           onesChannel(N);
  for(int channelId=0; channelId<N; channelId++)
    onesChannel.set(channelId, 1.);
  
  trafo.gemv(BLAS_OP_T,
             &one, &SM,
             &onesChannel,
             &zero, &s);

  /* Reconstruct density */
  for(int iteration=0; iteration<50; iteration++)
  {
    trafo.gemv(BLAS_OP_N,
               &one, &SM,
               &x,
               &zero, &e);

    trafo.divides(&y, &e, &yy);

    trafo.gemv(BLAS_OP_T,
               &one, &SM,
               &yy,
               &zero, &c);
    
    trafo.corrects(&x, &c, &s, &x);
  }
  
  /* Print */
  std::cout << std::endl
            << "Guess: | True:" << std::endl
            << "--------------" << std::endl;
  for(int voxelId=0; voxelId<M; voxelId++)
    std::cout << std::setw(7) << x.get(voxelId)
              << std::setw(8) << xx.get(voxelId)
              << std::endl;
  
  return 0;
}
