/* Reconstruction program for real measurement data.  The reconstruction method
 * is Summed Backprojection.
 */
#include "real_measurementsetup_defines.h"
#include "voxelgrid52_defines.h"

#include "CUDA_HandleError.hpp"
#include "FileTalk.hpp"

#include "ChordsCalc_kernel3.cu"
#include "MeasurementSetup.hpp"
#include "VoxelGrid.hpp"
#include "CudaMS.hpp"
#include "WriteableCudaVG.hpp"
#include "CudaTransform.hpp"
#include "H5Reader.hpp"
#include "H5DensityWriter.hpp"
#include "visualization.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include "MeasurementEvent.hpp"


#ifdef VERBOSE
#define VSAYLINE( x )     { SAYLINE( x); }
#define VSAYLINES( x, y ) { SAYLINES(x,y); }
#else
#define VSAYLINE( x )     {}
#define VSAYLINES( x, y ) {}
#endif


#define UPPERCHUNKID 1

typedef float val_t;

int main( int ac, char ** av )
{
  /* ---------------------------
   * Treat commandline arguments 
   * --------------------------- */
  VSAYLINES(__LINE__-3, __LINE__-1);
  
  if(ac < 5)
  {
    std::cerr << "Wrong number of arguments. Exspected arguments:" << std::endl
              << "    1.: measurement filename (mandatory)" << std::endl
              << "    2.: chunkSize (mandatory)" << std::endl
              << "    3.: randomSeed (mandatory)" << std::endl
              << "    4.: nThreadRays (mandatory)" << std::endl
              << "    5.: nThreadsPerBlock (mandatory)" << std::endl
              << "    6.: file output prefix (optional, defaults to \"real_algo_output\")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string fn(av[1]);
  int const   chunkSize(atoi(av[2]));
  int const   randomSeed(atoi(av[3]));
  int const   nThreadRays(atoi(av[4]));
  int const   nThreadsPerBlock(atoi(av[5]));

  std::string outpre;
  if(ac >= 7)
    outpre = std::string(av[6]);
  else
    outpre = std::string("real_algo_output");
  

  /* --------------
   * Create objects
   * -------------- */
  VSAYLINES(__LINE__-3, __LINE__-1);
  
  /* Voxel grid */
  WriteableCudaVG<val_t, DefaultVoxelGrid<val_t> > *
                            grid =

        new WriteableCudaVG<val_t, DefaultVoxelGrid<val_t> >(
              GRIDOX, GRIDOY, GRIDOZ,
              GRIDDX, GRIDDY, GRIDDZ,
              GRIDNX, GRIDNY, GRIDNZ);

  /* Measurement setup */
  CudaMS<val_t, DefaultMeasurementSetup<val_t> > *
                            setup =
        
        new CudaMS<val_t, DefaultMeasurementSetup<val_t> >(
              POS0X, POS1X,
              NA, N0Z, N0Y, N1Z, N1Y,
              DA, SEGX, SEGY, SEGZ);
  
  /* Transform (math object) */
  CudaTransform<val_t,val_t>
                            trafo;
 
  /* System matrix chunk */
#ifdef WITH_CUDAMATRIX
  CudaMatrix<val_t,val_t>   chunk(chunkSize, VGRIDSIZE);
  
  for(int rowId=0; rowId<chunkSize; rowId++)
    for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
      chunk.set(rowId, vxlId, 0.);
#else
  val_t * chunk_host = 0;
  chunk_host = (val_t*)malloc(chunkSize*VGRIDSIZE*sizeof(val_t));
  val_t * chunk_devi = 0;
  HANDLE_ERROR( cudaMalloc((void**)&chunk_devi,
                           chunkSize*VGRIDSIZE*sizeof(val_t)) );

  for(int rowId=0; rowId<chunkSize; rowId++)
    for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
      chunk_host[vxlId*chunkSize+rowId]=0.;
  HANDLE_ERROR( cudaMemcpy(chunk_devi, chunk_host,
                           chunkSize*VGRIDSIZE*sizeof(val_t),
                           cudaMemcpyHostToDevice) );
#endif
  
  /* Measurement vector */
  CudaVector<val_t, val_t>  yValues_chunk(chunkSize);

  CudaVector<MeasurementEvent<val_t>, MeasurementEvent<val_t> > 
                            y_chunk(chunkSize); // chunk part of meas.

  for(int listId=0; listId<chunkSize; listId++)
    y_chunk.set(listId, MeasurementEvent<val_t>(0., -1));
  
  /* Density guess */
  CudaVector<val_t,val_t>   x(VGRIDSIZE);

  for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
    x.set(vxlId, 0.);
  
  /* Helper */
  val_t one(1.);
  val_t zero(0.);
 
  
  /* ----------------
   * Read measurement
   * ---------------- */
  VSAYLINES(__LINE__-3, __LINE__-1);
  
  std::cout << "Total number of channels:" << std::endl
            << "    " << NCHANNELS << std::endl;
  
  /* Open measurement file */
  H5Reader h5reader(fn);
  
  /* Allocate memory for and read raw input data */
  if(!h5reader.is_open())
  {
    std::cerr << "Could not open file '" << fn << "'." << std::endl;
    exit(EXIT_FAILURE);
  }
  val_t * meas = new val_t[NCHANNELS];
  h5reader.read(meas);
  
  /* Count those channels, that have values != 0. */
  int count(0);
  for(int cnlId=0; cnlId<NCHANNELS; cnlId++)
    if(meas[cnlId] != 0.)
      count++;

  int const NEVENTS(count);
  int const NCHUNKS((NEVENTS+chunkSize-1)/chunkSize);
  
  std::cout << "Total number of events (non-zero channel values): " << std::endl
            << "    " << NEVENTS << std::endl;
  
  /* Create measurement vector */
  CudaVector<MeasurementEvent<val_t>, MeasurementEvent<val_t> >
                            y(NEVENTS);

  int listId(0);
  for(int cnlId=0; cnlId<NCHANNELS; cnlId++)
  {
    if(meas[cnlId] != 0.)
    {
       y.set(listId, MeasurementEvent<val_t>(meas[cnlId], cnlId));
       listId++;
    }
  }

#ifdef DEBUG  
/**//* Print measurement vector */
/**/SAYLINE(__LINE__-1);
/**/std::cout << "y:"
/**/          << std::endl;
/**/for(int listId=0; listId<NEVENTS; listId++)
/**/{
/**/  MeasurementEvent<val_t> event = y.get(listId);
/**/  std::stringstream ss("");
/**/  ss << "listId " << listId << ": ("
/**/     << event.channel() << ": " << event.value() << ")";
/**/  std::cout << std::right
/**/            << std::setw(15) << ss.str() << " "
/**/            << std::endl;
/**/}
#endif  // DEBUG


  /* ----------------
   * Reconstruct
   * ---------------- */
  VSAYLINES(__LINE__-3, __LINE__-1);

  /* Iterate over chunks */
  VSAYLINE(__LINE__-1);
  //for(int chunkId=0; (chunkId<UPPERCHUNKID) && (chunkId<NCHUNKS); chunkId++)
  for(int chunkId=0; chunkId<NCHUNKS; chunkId++)
  {
    /* Copy chunk's part of measurement vector */
    VSAYLINE(__LINE__-1);
    
    for(int listId=0; listId<chunkSize; listId++)
    {
      MeasurementEvent<val_t> event;
      if(chunkId*chunkSize + listId < NEVENTS)
        event = y.get(chunkId*chunkSize + listId);
      else
        event = MeasurementEvent<val_t>(0., -1);

      assert(!isnan(event.value()));
      assert(!isinf(event.value()));

      y_chunk.set(      listId, event);
      yValues_chunk.set(listId, event.value());
    }
    
#ifdef DEBUG 
/**//* Print measurement vector */
/**/SAYLINE(__LINE__-1);
/**/std::cout << std::left
/**/          << std::setw(16) << "y_chunk:"
/**/          << std::setw(16) << "yValues_chunk:"
/**/          << std::endl;
/**/for(int listId=0; listId<chunkSize; listId++)
/**/{
/**/  MeasurementEvent<val_t> event = y_chunk.get(listId);
/**/  val_t                   elem  = yValues_chunk.get(listId);
/**/  std::stringstream ss("");
/**/  ss << "(" << event.channel() << ": " << event.value() << ")";
/**/  std::cout << std::right
/**/            << std::setw(15) << ss.str() << " "
/**/            << std::setw(15) << elem     << " "
/**/            << std::endl;
/**/}
#endif  // DEBUG
    
    /* Set system matrix chunk's elements to null */
    VSAYLINE(__LINE__-1);
#ifdef WITH_CUDAMATRIX
    for(int listId=0; listId<chunkSize; listId++)
      for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
        chunk.set(listId, vxlId, 0.);
    HANDLE_ERROR( cudaDeviceSynchronize() );
#else
    for(int listId=0; listId<chunkSize; listId++)
      for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
        chunk_host[vxlId*chunkSize+listId]=0.;
    HANDLE_ERROR( cudaMemcpy(chunk_devi, chunk_host,
                             chunkSize*VGRIDSIZE*sizeof(val_t),
                             cudaMemcpyHostToDevice) );
#endif

    /* Calculate system matrix chunk */
    VSAYLINE(__LINE__-1);
    chordsCalc_noVis(
          chunkId, NCHANNELS, chunkSize, nThreadsPerBlock,
#ifdef WITH_CUDAMATRIX
          static_cast<val_t*>(chunk.data()),
#else
          chunk_devi,
#endif
          &y_chunk,
          grid,
          VGRIDSIZE,
          setup,
          randomSeed,
          nThreadRays);
    HANDLE_ERROR( cudaDeviceSynchronize() );
#ifdef WITH_CUDAMATRIX
    chunk.set_devi_data_changed();
#endif

//#ifndef WITH_CUDAMATRIX    
//    HANDLE_ERROR( cudaMemcpy(chunk_host, chunk_devi,
//                             chunkSize*VGRIDSIZE*sizeof(val_t),
//                             cudaMemcpyDeviceToHost) );
//#endif
//    for(int listId=0; listId<chunkSize; listId++)
//    {
//      for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
//      {
//#ifdef WITH_CUDAMATRIX
//        val_t elem = chunk.get(listId, vxlId);
//#else
//        val_t elem = chunk_host[vxlId*chunkSize+listId];
//#endif
//        assert(!isnan(elem));
//        assert(!isinf(elem));
//      }
//    }

#ifdef DEBUG
/**//* Print system matrix chunk */
/**/SAYLINE(__LINE__-1);
/**/std::cout << "chunk:" << std::endl;
/**/for(int listId=0; listId<chunkSize; listId++)
/**/{
/**/  int count(0);
/**/  for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
/**/    //if(chunk.get(listId, vxlId) != 0.) count++;
/**/    if(chunk_host[vxlId*chunkSize+listId] != 0.) count++;
/**/
/**/  if(count > 0)
/**/  {
/**/    std::cout << "  listId " << listId << ":  ";
/**/    for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
/**/    {
/**/      //val_t elem = chunk.get(listId, vxlId);
/**/      val_t elem = chunk_host[vxlId*chunkSize+listId];
/**/      if(elem != 0.)
/**/        std::cout << elem << "  ";
/**/    }
/**/    std::cout << std::endl;
/**/  }
/**/}
#endif  // DEBUG

#ifdef WITH_CUDAMATRIX
    /* Back projection */
    VSAYLINE(__LINE__-1);
    trafo.gemv(
          BLAS_OP_T,
          &one, &chunk,
          &yValues_chunk,
          &one, &x);
    x.set_devi_data_changed();
#endif

#ifdef DEBUG
/**//* Print x */
/**/SAYLINE(__LINE__-1);
/**/std::cout << "x:" << std::endl;
/**/for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
/**/  if(x.get(vxlId) != 0.)
/**/    std::cout << "  " << x.get(vxlId);
/**/std::cout << std::endl;
#endif  // DEBUG
  } /* End iterate over chunks */


  /* ----------------
   * File output
   * ---------------- */
  VSAYLINES(__LINE__-3, __LINE__-1);
  
  /* Write last guess */
  VSAYLINE(__LINE__-1);
  val_t * guess = new val_t[VGRIDSIZE];
  for(int memid=0; memid<VGRIDSIZE; memid++)
    guess[memid] = x.get(memid);

  H5DensityWriter<WriteableCudaVG<val_t, DefaultVoxelGrid<val_t> > >
        h5writer(outpre + std::string("_x.h5"));
  
  h5writer.write(guess, *grid);
  
  /* Visualize grid */
  std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
  VSAYLINE(__LINE__-1);
  DefaultVoxelGrid<val_t> * hostRepr = grid->hostRepr();
  VSAYLINE(__LINE__-1);
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
  PlyWriter writer(outpre + std::string("_grid.ply"));
  writer.write(visGrid);
  writer.close();

  /* Visualize det0 */
  int   det0N[] = {1, N0Y, N0Z};
  val_t det0C[] = {POS0X, 0, 0};
  val_t detD[]  = {SEGX, SEGY, SEGZ};
  BetaPlyGrid<val_t> det0(
        "", det0C, detD, det0N, BetaPlyGrid<val_t>::AT_CENTER);
  PlyWriter det0Writer(outpre + std::string("_det0.ply"));
  det0Writer.write(det0);
  det0Writer.close();

  /* Visualize det1 */
  int   det1N[] = {1, N1Y, N1Z};
  val_t det1C[] = {POS1X, 0, 0};
  BetaPlyGrid<val_t> det1("", det1C, detD, det1N, BetaPlyGrid<val_t>::AT_CENTER);
  PlyWriter det1Writer(outpre + std::string("_det1.ply"));
  det1Writer.write(det1);
  det1Writer.close();



  return 0;
}
