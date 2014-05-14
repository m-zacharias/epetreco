/* Reconstruction development example.  Uses a small and simple 2D geometry.
 * 
 * For the chosen setup (detectors, voxel grid), a true density vector xx is
 * defined.  Using the system matrix, its "true" measurement vector y is
 * calculated.  The vector that holds the density guess is x, its elements are
 * initialized with a constant value.  This guess is then iteratively improved
 * using MLEM.
 */

#include "FileTalk.hpp"
#include "CUDA_HandleError.hpp"
#include <iomanip>

#include "defines.h"
#include "ChordsCalc_kernel2.cu"
#include "MeasurementSetup.hpp"
#include "VoxelGrid.hpp"
#include "CudaTransform.hpp"
#include "visualization.hpp"

#define M VGRIDSIZE // number of voxels
#define N NCHANNELS // number of channels
#define NITERATIONS 50



template<typename T, typename ConcreteVoxelGrid>
class CudaVG
{
  public:
    
    CudaVG( T const   gridO0, T const   gridO1, T const   gridO2,
            T const   gridD0, T const   gridD1, T const   gridD2,
            int const gridN0, int const gridN1, int const gridN2 )
    {
      // Allocate host memory
      _data_host = new ConcreteVoxelGrid(gridO0, gridO1, gridO2,
                                         gridD0, gridD1, gridD2,
                                         gridN0, gridN1, gridN2);
      _host_data_changed = true;
      
      // Allocate device memory
      cudaError_t status;
      status =
            cudaMalloc((void**)&_data_devi, sizeof(ConcreteVoxelGrid));
      if(status != cudaSuccess)
      {
        std::cerr << "CudaVG::CudaVG(...) : cudaMalloc(...) failed" << std::endl;
        exit(EXIT_FAILURE);
      }
      _devi_data_changed = false;
    }

    ~CudaVG()
    {
      delete _data_host;
      cudaFree(_data_devi);
    }

    ConcreteVoxelGrid * deviRepr()
    {
      if(_host_data_changed)
        update_devi_data();
      
      _devi_data_changed = true;
      return _data_devi;
    }

    ConcreteVoxelGrid * hostRepr()
    {
      if(_devi_data_changed)
        update_host_data();
      
      _host_data_changed = true;
      return _data_host;
    }


  private:
    
    ConcreteVoxelGrid * _data_host;
    
    ConcreteVoxelGrid * _data_devi;

    bool _host_data_changed;

    bool _devi_data_changed;

    void update_host_data()
    {
      cudaError_t status;
      status =
            cudaMemcpy(_data_host, _data_devi, sizeof(ConcreteVoxelGrid),
                       cudaMemcpyDeviceToHost);
      if(status != cudaSuccess)
      {
        std::cerr << "CudaVG::update_host_data() : cudaMemcpy(...) failed"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      _devi_data_changed = false;
    }

    void update_devi_data()
    {
      cudaError_t status;
      status =
            cudaMemcpy(_data_devi, _data_host, sizeof(ConcreteVoxelGrid),
                       cudaMemcpyHostToDevice);
      if(status != cudaSuccess)
      {
        std::cerr << "CudaVG::update_devi_data() : cudaMemcpy(...) failed"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      _host_data_changed = false;
    }
};



template<typename T, typename ConcreteMeasurementSetup>
class CudaMS
{
  public:
    
    CudaMS(
          T   pos0x, T   pos1x,
          int na,    int n0z,   int n0y,  int n1z, int n1y,
          T   da,    T   segx,  T   segy, T   segz )
    {
      // Allocate host memory
      _data_host = new ConcreteMeasurementSetup(
            pos0x, pos1x, na, n0z, n0y, n1z, n1y, da, segx, segy, segz);
      _host_data_changed = true;
      
      // Allocate device memory
      cudaError_t status;
      status =
            cudaMalloc((void**)&_data_devi, sizeof(ConcreteMeasurementSetup));
      _devi_data_changed = false; 
      if(status != cudaSuccess)
      {
        std::cerr << "CudaMS::CudaMS(...) : cudaMalloc(...) failed" << std::endl;
        exit(EXIT_FAILURE);
      }
      _devi_data_changed = false;
    }

    ~CudaMS()
    {
      delete _data_host;
      cudaFree(_data_devi);
    }

    ConcreteMeasurementSetup * deviRepr()
    {
      if(_host_data_changed)
        update_devi_data();
      
      _devi_data_changed = true;
      return _data_devi;
    }

    ConcreteMeasurementSetup * hostRepr()
    {
      if(_devi_data_changed)
        update_host_data();
      
      _host_data_changed = true;
      return _data_host;
    }


  private:
    
    ConcreteMeasurementSetup * _data_host;
    
    ConcreteMeasurementSetup * _data_devi;

    bool _host_data_changed;

    bool _devi_data_changed;

    void update_host_data()
    {
      cudaError_t status;
      status =
            cudaMemcpy(_data_host, _data_devi, sizeof(ConcreteMeasurementSetup),
                       cudaMemcpyDeviceToHost);
      if(status != cudaSuccess)
      {
        std::cerr << "CudaMS::update_host_data() : cudaMemcpy(...) failed"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      _devi_data_changed = false;
    }

    void update_devi_data()
    {
      cudaError_t status;
      status =
            cudaMemcpy(_data_devi, _data_host, sizeof(ConcreteMeasurementSetup),
                       cudaMemcpyHostToDevice);
      if(status != cudaSuccess)
      {
        std::cerr << "CudaMS::update_devi_data() : cudaMemcpy(...) failed"
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
      CudaVG<float, DefaultVoxelGrid<float> > * const grid,
      int const vgridSize,
      CudaMS<float, DefaultMeasurementSetup<float> > * const setup )
{
  chordsCalc<<<chunkSize, nThreads>>>(chords, rays,
                                      grid->deviRepr()->gridO,
                                      grid->deviRepr()->gridD,
                                      grid->deviRepr()->gridN,
                                      chunkId*chunkSize, nChannels, chunkSize, vgridSize,
                                      setup->deviRepr());
}



void chordsCalc_noVis(
      int const chunkId, int const nChannels, int const chunkSize, int const nThreads,
      float * const chords,
      CudaVG<float, DefaultVoxelGrid<float> > * const grid,
      int const vgridSize,
      CudaMS<float, DefaultMeasurementSetup<float> > * const setup )
{
  chordsCalc_noVis<<<chunkSize, nThreads>>>(chords,
                                            grid->deviRepr()->gridO,
                                            grid->deviRepr()->gridD,
                                            grid->deviRepr()->gridN,
                                            chunkId*chunkSize, nChannels,
                                            chunkSize, vgridSize,
                                            setup->deviRepr());
}



typedef float val_t;

int main()
{
  /* ###########################################################################
   * ### CALCULATE MEASUREMENT VECTOR
   * ######################################################################## */
  
  /* Create objects */
  SAYLINE(__LINE__-1);
  CudaVG<val_t, DefaultVoxelGrid<val_t> > * grid =
        new CudaVG<val_t, DefaultVoxelGrid<val_t> >(
              -1.5,    -0.5,  -2.0,
              1.0,     1.0,   1.0,
              GRIDNX, GRIDNY, GRIDNZ);
  CudaMS<val_t, DefaultMeasurementSetup<val_t> > * setup =
        new CudaMS<val_t, DefaultMeasurementSetup<val_t> >(
                POS0X, POS1X, NA, N0Z, N0Y, N1Z, N1Y,
                DA, SEGX, SEGY, SEGZ);
  CudaTransform<val_t,val_t>        trafo;
  CudaDeviceOnlyMatrix<val_t,val_t> SM(N, M);       // system matrix
  CudaVector<val_t,val_t>           xx(M);          // true density
  for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
  {
    xx.set(voxelId, 0.);
    if(voxelId == 9 || voxelId == 3)
      xx.set(voxelId, 1.);
  }
  CudaVector<val_t,val_t>           y(N);           // true measurement
  val_t one(1.);
  val_t zero(0.);
  
  /* Calculate system matrix */
  SAYLINE(__LINE__-1);
  chordsCalc_noVis(0, N, N, 1, static_cast<val_t*>(SM.data()), grid,
                   VGRIDSIZE, setup);
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
  
  int const CHUNKSIZE(NCHANNELS/20);
  int const NCHUNKS((NCHANNELS+CHUNKSIZE-1)/CHUNKSIZE);

  /* Create objects */
  CudaDeviceOnlyMatrix<val_t,val_t> chunk(CHUNKSIZE, M);// system matrix
  CudaVector<val_t,val_t>           e(CHUNKSIZE);       // simulated measurement
  CudaVector<val_t,val_t>           y_chunk(CHUNKSIZE); // part of true meas vct
  CudaVector<val_t,val_t>           yy(CHUNKSIZE);      // "error"
  CudaVector<val_t,val_t>           c(M);               // correction
  CudaVector<val_t,val_t>           s(M);               // sensitivity
  CudaVector<val_t,val_t>           x(M);               // density guess
  for(int voxelId=0; voxelId<M; voxelId++)
    x.set(voxelId, 1.);

  /* ---------------------
   * Calculate sensitivity
   * --------------------- */
  SAYLINES(__LINE__-3, __LINE__-1);
  /* Set sensitivity elements to null */
  for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
    s.set(voxelId, 0.);
  
  /* Create vector of ones of chunk's size */
  CudaVector<val_t,val_t>           onesChannel(CHUNKSIZE);
  for(int channelId=0; channelId<CHUNKSIZE; channelId++)
    onesChannel.set(channelId, 1.);
  
  for(int chunkId=0; chunkId<NCHUNKS; chunkId++)        // for chunks
  {
    /* Set system matrix chunk elements to null */
    for(int i=0; i<CHUNKSIZE; i++)
      for(int j=0; j<VGRIDSIZE; j++)
        chunk.set(i, j, 0.);
    HANDLE_ERROR( cudaDeviceSynchronize() );
    
    /* Calculate system matrix chunk */
    chordsCalc_noVis(chunkId, NCHANNELS, CHUNKSIZE, 1,
                     static_cast<val_t*>(chunk.data()), grid,
                     VGRIDSIZE, setup);
    HANDLE_ERROR( cudaDeviceSynchronize() );
    
    /* Add column sums to sensitivity */
    trafo.gemv(BLAS_OP_T,
               &one, &chunk,
               &onesChannel,
               &one, &s);
  }
  
  /* Print sensitivity */
  std::cout << std::endl
            << "s: ";
  for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
    std::cout << std::setw(9) << s.get(voxelId);
  std::cout << std::endl;
  
  /* ----------
   * Iterations
   * ---------- */

  /* Allocate memory for rays */
  val_t rays_host[NCHUNKS*CHUNKSIZE*NTHREADRAYS*6*sizeof(val_t)];
  val_t * rays_devi;
  HANDLE_ERROR( cudaMalloc((void**)&rays_devi,
                NCHUNKS*CHUNKSIZE*NTHREADRAYS*6*sizeof(val_t)) );
  
  for(int iteration=0; iteration<NITERATIONS; iteration++)  // for iterations
  {
    std::cout << "Iteration " << iteration << std::endl
              << "------------" << std::endl;

    /* Set correction images elements to null */
    for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
      c.set(voxelId, 0.);
    
    for(int chunkId=0; chunkId<NCHUNKS; chunkId++)      // for chunks
    {
      std::cout << "/ ****" << std::endl
                << "| chunkId: " << chunkId << std::endl;
      
      /* Copy chunk's part of measurement vector */
      for(int channelId=0; channelId<CHUNKSIZE; channelId++)
        if(chunkId*CHUNKSIZE+channelId<NCHANNELS)
          y_chunk.set(channelId, y.get(chunkId*CHUNKSIZE+channelId));
        else
          y_chunk.set(channelId, 0.);
      
      /* Set system matrix chunk's elements to null */
      for(int i=0; i<CHUNKSIZE; i++)
        for(int j=0; j<VGRIDSIZE; j++)
          chunk.set(i, j, 0.);
      HANDLE_ERROR( cudaDeviceSynchronize() );
      
      /* Calculate system matrix chunk */
      chordsCalc(chunkId, NCHANNELS, CHUNKSIZE, 1,
                 static_cast<val_t*>(chunk.data()),
//                 &rays_devi[chunkId*CHUNKSIZE*NTHREADRAYS*6], grid,
                 rays_devi, grid,
                 VGRIDSIZE, setup);
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
      
      /* Calculate simulated measurement of current density guess */
      trafo.gemv(BLAS_OP_N,
                 &one, &chunk,
                 &x,
                 &zero, &e);

      /* Calculate "error" */
      trafo.divides(&y_chunk, &e, &yy);
      
      for(int channelId=0; channelId<CHUNKSIZE; channelId++)
        if(e.get(channelId) == 0)
          yy.set(channelId, 0.);
      
      /* Add back transformed error to correction image */
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
  DefaultVoxelGrid<val_t> * hostRepr = grid->hostRepr();
  PlyGrid<TemplateVertex<val_t> > visGrid("",
                          TemplateVertex<val_t>(hostRepr->gridO[0],
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
  BetaPlyGrid<val_t> det0(
        "", det0C, detD, det0N, BetaPlyGrid<val_t>::AT_CENTER);
  PlyWriter det0Writer("ChordsCalc_kernel2_det0.ply");
  det0Writer.write(det0);
  det0Writer.close();

  // Visualize det1
  int   det1N[] = {1, N1Y, N1Z};
  val_t det1C[] = {POS1X, 0, 0};
  BetaPlyGrid<val_t> det1("", det1C, detD, det1N, BetaPlyGrid<val_t>::AT_CENTER);
  PlyWriter det1Writer("ChordsCalc_kernel2_det1.ply");
  det1Writer.write(det1);
  det1Writer.close();
  
  // Visualize rays
  HANDLE_ERROR( cudaMemcpy(rays_host, rays_devi,
                           6*NCHUNKS*CHUNKSIZE*NTHREADRAYS*sizeof(val_t),
                           cudaMemcpyDeviceToHost) );
  for(int i=0; i<NBLOCKS; i++)
  {
    BetaCompositePlyGeom compositeLines("");
    BetaPlyLine<val_t> lines[NTHREADRAYS];
    for(int idRay=0; idRay<NTHREADRAYS; idRay++)
    {
      lines[idRay] = BetaPlyLine<val_t>("", &rays_host[6*(i*NTHREADRAYS + idRay)]);
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
