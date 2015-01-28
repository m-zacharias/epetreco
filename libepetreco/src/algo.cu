/** @file algo.cu */
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
#include "CudaVG.hpp"
#include "CudaMS.hpp"
//#include "ChordsCalc_kernelWrapper.hpp"

#define M VGRIDSIZE // number of voxels
#define N NCHANNELS // number of channels
#define NITERATIONS 50



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
  for(int cnlId=0; cnlId<N; cnlId++)
    for(int vxlId=0; vxlId<M; vxlId++)
      SM.set(cnlId, vxlId, 0.);

  CudaVector<val_t,val_t>           xx(M);          // true density
  for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
  {
    xx.set(voxelId, 0.);
    if(voxelId == 9 || voxelId == 3)
      xx.set(voxelId, 1.);
  }
  
  CudaVector<val_t,val_t>           y(N);           // true measurement
  for(int cnlId=0; cnlId<N; cnlId++)
    y.set(cnlId, 0.);

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
  for(int cnlId=0; cnlId<CHUNKSIZE; cnlId++)
    for(int vxlId=0; vxlId<M; vxlId++)
      chunk.set(cnlId, vxlId, 0.);

  CudaVector<val_t,val_t>           e(CHUNKSIZE);       // simulated measurement
  for(int cnlId=0; cnlId<CHUNKSIZE; cnlId++)
    e.set(cnlId, 0.);

  CudaVector<val_t,val_t>           y_chunk(CHUNKSIZE); // part of true meas vct
  for(int cnlId=0; cnlId<CHUNKSIZE; cnlId++)
    y_chunk.set(cnlId, 0.);
  
  CudaVector<val_t,val_t>           yy(CHUNKSIZE);      // "error"
  for(int cnlId=0; cnlId<CHUNKSIZE; cnlId++)
    yy.set(cnlId, 0.);
  
  CudaVector<val_t,val_t>           c(M);               // correction
  for(int vxlId=0; vxlId<CHUNKSIZE; vxlId++)
    c.set(vxlId, 0.);
  
  CudaVector<val_t,val_t>           s(M);               // sensitivity
  for(int vxlId=0; vxlId<CHUNKSIZE; vxlId++)
    s.set(vxlId, 0.);
  
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
