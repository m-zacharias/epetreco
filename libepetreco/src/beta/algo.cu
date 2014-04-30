/* Reconstruction development example.  Uses a small and simple 2D geometry.
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
#include "FileTalk.hpp"
#include "visualization.hpp"

#define M VGRIDSIZE // number of voxels
#define N NCHANNELS // number of channels
#define NITERATIONS 50

struct BaseGrid
{
  BaseGrid( float const gridO0, float const gridO1, float const gridO2,
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

class Grid
{
  public:
    
    Grid( float const gridO0, float const gridO1, float const gridO2,
          float const gridD0, float const gridD1, float const gridD2,
          int const   gridN0, int const   gridN1, int const   gridN2 )
    {
      // Allocate host memory
      _data_host = new BaseGrid(gridO0, gridO1, gridO2,
                                gridD0, gridD1, gridD2,
                                gridN0, gridN1, gridN2);
      _host_data_changed = true;
      
      // Allocate device memory
      cudaError_t status;
      status =
            cudaMalloc((void**)&_data_devi, sizeof(BaseGrid));
      if(status != cudaSuccess)
      {
        std::cerr << "Grid::Grid(...) : cudaMalloc(...) failed" << std::endl;
        exit(EXIT_FAILURE);
      }
      _devi_data_changed = false;
    }

    ~Grid()
    {
      delete _data_host;
      cudaFree(_data_devi);
    }

    BaseGrid * deviRepr()
    {
      if(_host_data_changed)
        update_devi_data();
      
      _devi_data_changed = true;
      return _data_devi;
    }

    BaseGrid * hostRepr()
    {
      if(_devi_data_changed)
        update_host_data();
      
      _host_data_changed = true;
      return _data_host;
    }


  private:
    
    BaseGrid * _data_host;
    
    BaseGrid * _data_devi;

    bool _host_data_changed;

    bool _devi_data_changed;

    void update_host_data()
    {
      cudaError_t status;
      status =
            cudaMemcpy(_data_host, _data_devi, sizeof(BaseGrid),
                       cudaMemcpyDeviceToHost);
      if(status != cudaSuccess)
      {
        std::cerr << "Grid::update_host_data() : cudaMemcpy(...) failed"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      _devi_data_changed = false;
    }

    void update_devi_data()
    {
      cudaError_t status;
      status =
            cudaMemcpy(_data_devi, _data_host, sizeof(BaseGrid),
                       cudaMemcpyHostToDevice);
      if(status != cudaSuccess)
      {
        std::cerr << "Grid::update_devi_data() : cudaMemcpy(...) failed"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      _host_data_changed = false;
    }
};

void chordsCalc(
      int const chunkId, int const nChannels, int const chunkSize, int const nThreads,
      float * const chords,
      float * const rays,
      Grid * const grid,
      int const vgridSize )
{
  chordsCalc<<<chunkSize, nThreads>>>(chords, rays,
                                      grid->deviRepr()->gridO,
                                      grid->deviRepr()->gridD,
                                      grid->deviRepr()->gridN,
                                      chunkId*chunkSize, nChannels, chunkSize, vgridSize);
}





int main()
{
  /* ###########################################################################
   * ### CALCULATE MEASUREMENT VECTOR
   * ######################################################################## */
  
  /* Create objects */
  SAYLINE(__LINE__-1);
  Grid * grid =
        new Grid(-1.5,    -0.5,  -2.0,
                  1.0,     1.0,   1.0,
                  GRIDNX, GRIDNY, GRIDNZ);
  CudaTransform<float,float>        trafo;
  CudaDeviceOnlyMatrix<float,float> SM(N, M);       // system matrix
  CudaVector<float,float>           xx(M);          // true density
  for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
  {
    xx.set(voxelId, 0.);
    if(voxelId == 9 || voxelId == 3)
      xx.set(voxelId, 1.);
  }
  CudaVector<float,float>           y(N);           // true measurement
  float one(1.);
  float zero(0.);
  float rays_host[6*N*NTHREADRAYS*sizeof(float)];
  float * rays_devi;
  HANDLE_ERROR( cudaMalloc((void**)&rays_devi, 6*N*NTHREADRAYS*sizeof(float)) );
  
  /* Calculate system matrix */
  SAYLINE(__LINE__-1);
  chordsCalc(0, N, N, 1, static_cast<float*>(SM.data()), rays_devi, grid,
             VGRIDSIZE);
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

  /* Calculate "true" measurement */
  SAYLINE(__LINE__-1);
  trafo.gemv(BLAS_OP_N,
             &one, &SM,
             &xx,
             &zero, &y);
  
  std::cout << std::endl
            << "\"True\" measurement vector:" << std::endl
            << "----------------------------" << std::endl;
  for(int channelId=0; channelId<N; channelId++)
    std::cout << y.get(channelId) << std::endl;
  

  /* ###########################################################################
   * ### RECONSTRUCT
   * ######################################################################## */
  SAYLINES(__LINE__-3, __LINE__-1);
  
  int const CHUNKSIZE(NCHANNELS/3);
  int const NCHUNKS((NCHANNELS+CHUNKSIZE-1)/CHUNKSIZE);

  /* Create objects */
  CudaDeviceOnlyMatrix<float,float> chunk(CHUNKSIZE, M);// system matrix
  CudaVector<float,float>           e(CHUNKSIZE);       // simulated measurement
  CudaVector<float,float>           y_chunk(CHUNKSIZE); // part of true meas vct
  CudaVector<float,float>           yy(CHUNKSIZE);      // "error"
  CudaVector<float,float>           c(M);               // correction
  CudaVector<float,float>           s(M);               // sensitivity
  CudaVector<float,float>           x(M);               // density guess
  for(int voxelId=0; voxelId<M; voxelId++)
    x.set(voxelId, 1.);

  /* Calculate sensitivity */
  SAYLINE(__LINE__-1);
  for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)      // set to zero
    s.set(voxelId, 0.);

  CudaVector<float,float>           onesChannel(CHUNKSIZE);
  for(int channelId=0; channelId<CHUNKSIZE; channelId++)
    onesChannel.set(channelId, 1.);
  
  for(int chunkId=0; chunkId<NCHUNKS; chunkId++)
  {
    /* Calculate system matrix */
    for(int i=0; i<CHUNKSIZE; i++)
      for(int j=0; j<VGRIDSIZE; j++)
        chunk.set(i, j, 0.);
    HANDLE_ERROR( cudaDeviceSynchronize() );
    chordsCalc(chunkId, NCHANNELS, CHUNKSIZE, 1,
               static_cast<float*>(chunk.data()), rays_devi, grid,
               VGRIDSIZE);
    HANDLE_ERROR( cudaDeviceSynchronize() );
    
    trafo.gemv(BLAS_OP_T,
               &one, &chunk,
               &onesChannel,
               &one, &s);
  }

  std::cout << std::endl
            << "s: ";
  for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
    std::cout << std::setw(9) << s.get(voxelId);
  std::cout << std::endl;
  
  /* Iterations */
  for(int iteration=0; iteration<NITERATIONS; iteration++)
  {
    std::cout << "Iteration " << iteration << std::endl
              << "------------" << std::endl;

    /* Set correction to zero */
    for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
      c.set(voxelId, 0.);
    
    for(int chunkId=0; chunkId<NCHUNKS; chunkId++)
    {
      std::cout << "/ ****" << std::endl
                << "| chunkId: " << chunkId << std::endl;
      
      /* Copy part of measurement vector */
      for(int channelId=0; channelId<CHUNKSIZE; channelId++)
        if(chunkId*CHUNKSIZE+channelId<NCHANNELS)
          y_chunk.set(channelId, y.get(chunkId*CHUNKSIZE+channelId));
        else
          y_chunk.set(channelId, 0.);

      /* Calculate system matrix */
      for(int i=0; i<CHUNKSIZE; i++)
        for(int j=0; j<VGRIDSIZE; j++)
          chunk.set(i, j, 0.);
      HANDLE_ERROR( cudaDeviceSynchronize() );
      chordsCalc(chunkId, NCHANNELS, CHUNKSIZE, 1,
                 static_cast<float*>(chunk.data()), rays_devi, grid,
                 VGRIDSIZE);
      HANDLE_ERROR( cudaDeviceSynchronize() );
      
      std::cout << "|" << std::endl
                << "| System matrix chunk:" << std::endl;
      std::cout << std::setprecision(2);
      for(int i=0; i<CHUNKSIZE; i++)
      {
        std::cout << "|   ";
        for(int j=0; j<M; j++)
          std::cout << std::setw(7) << chunk.get(i, j);
        std::cout << std::endl;
      }
      
      /* Calculate "error" */
      trafo.gemv(BLAS_OP_N,
                 &one, &chunk,
                 &x,
                 &zero, &e);

      trafo.divides(&y_chunk, &e, &yy);
      
      for(int channelId=0; channelId<CHUNKSIZE; channelId++)
        if(e.get(channelId) == 0)
          yy.set(channelId, 0.);
      
      /* Add up correction */
      trafo.gemv(BLAS_OP_T,
                 &one, &chunk,
                 &yy,
                 &one, &c);
      
      /* Print */
      std::cout   << "|" << std::endl
                  << "| "
                  << std::setw(7) << "y_chunk"
                  << std::setw(7) << "e"
                  << std::setw(7) << "yy" << std::endl;
      for(int channelId=0; channelId<CHUNKSIZE; channelId++)
        std::cout << "| "
                  << std::setw(7) << y_chunk.get(channelId)
                  << std::setw(7) << e.get(channelId)
                  << std::setw(7) << yy.get(channelId) << std::endl;

      std::cout   << "|" << std::endl
                  << "| "
                  << "c ";
      for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
        std::cout << std::setw(7) << c.get(voxelId);
      std::cout << std::endl;
      
      std::cout << "\\ ****" << std::endl;
    }
    
    /* Calculate new guess */
    trafo.corrects(&x, &c, &s, &x);
  }
  
  /* Print */
  SAYLINE(__LINE__-1);
  std::cout << std::endl
            << "Guess: | True:" << std::endl
            << "--------------" << std::endl;
  for(int voxelId=0; voxelId<M; voxelId++)
    std::cout << std::setw(7) << x.get(voxelId)
              << std::setw(8) << xx.get(voxelId)
              << std::endl;

  // Visualize grid
  BaseGrid * hostRepr = grid->hostRepr();
  PlyGrid<Vertex> visGrid("",
                          Vertex(hostRepr->gridO[0],
                                 hostRepr->gridO[1],
                                 hostRepr->gridO[2]),
                          hostRepr->gridN[0]+1,
                          hostRepr->gridN[1]+1,
                          hostRepr->gridN[2]+1,
                          hostRepr->gridD[0],
                          hostRepr->gridD[1],
                          hostRepr->gridD[2]);
  PlyWriter writer("algo_grid.ply");
  writer.write(visGrid);
  writer.close();

  // Visualize det0
  int   det0N[] = {1, N0Y, N0Z};
  val_t det0C[] = {POS0X, 0, 0};
  val_t detD[]  = {SEGX, SEGY, SEGZ};
  BetaPlyGrid det0("", det0C, detD, det0N, BetaPlyGrid::AT_CENTER);
  PlyWriter det0Writer("ChordsCalc_kernel2_det0.ply");
  det0Writer.write(det0);
  det0Writer.close();

  // Visualize det1
  int   det1N[] = {1, N1Y, N1Z};
  val_t det1C[] = {POS1X, 0, 0};
  BetaPlyGrid det1("", det1C, detD, det1N, BetaPlyGrid::AT_CENTER);
  PlyWriter det1Writer("ChordsCalc_kernel2_det1.ply");
  det1Writer.write(det1);
  det1Writer.close();
  
  // Visualize rays
  HANDLE_ERROR( cudaMemcpy(rays_host, rays_devi,
                           6*N*NTHREADRAYS*sizeof(val_t),
                           cudaMemcpyDeviceToHost) );
  for(int i=0; i<NBLOCKS; i++)
  {
    BetaCompositePlyGeom compositeLines("");
    BetaPlyLine lines[NTHREADRAYS];
    for(int idRay=0; idRay<NTHREADRAYS; idRay++)
    {
      lines[idRay] = BetaPlyLine("", &rays_host[6*(i*NTHREADRAYS + idRay)]);
      compositeLines.add(&lines[idRay]);
    }

    std::stringstream fn("");
    fn << "ChordsCalc_kernel2_rays-"
       << i << ".ply";
    PlyWriter raysWriter(fn.str());
    raysWriter.write(compositeLines);
    raysWriter.close();
  }
  
  return 0;
}
