/* Reconstruction program for real measurement data.  The reconstruction method
 * is Summed Backprojection.
 */

#ifndef MEASUREMENTSETUP_DEFINES
#define MEASUREMENTSETUP_DEFINES

#define N0Z 13        // 1st detector's number of segments in z
#define N0Y 13        // 1st detector's number of segments in y
#define N1Z 13        // 2nd detector's number of segments in z
#define N1Y 13        // 2nd detector's number of segments in y
#define NA  180       // number of angular positions
#define DA  2.        // angular step
#define POS0X -0.200  // position of 1st detector's center in x [m]
#define POS1X  0.700  // position of 2nd detector's center in x [m]
#define SEGX 0.02     // x edge length of one detector segment [m]
#define SEGY 0.004    // y edge length of one detector segment [m]
#define SEGZ 0.004    // z edge length of one detector segment [m]
#define NCHANNELS NA*N0Z*N0Y*N1Z*N1Y

#endif  // #define MEASUREMENTSETUP_DEFINES



#ifndef VOXELGRID_DEFINES
#define VOXELGRID_DEFINES

#define GRIDNX 4      // x dimension of voxel grid
#define GRIDNY 4      // y dimension of voxel grid
#define GRIDNZ 4      // z dimension od voxel grid
#define GRIDOX -0.05  // x origin of voxel grid [m]
#define GRIDOY -0.05  // y origin of voxel grid [m]
#define GRIDOZ -0.05  // z origin of voxel grid [m]
#define GRIDDX  0.025 // x edge length of one voxel [m]
#define GRIDDY  0.025 // y edge length of one voxel [m]
#define GRIDDZ  0.025 // z edge length of one voxel [m]
#define VGRIDSIZE GRIDNX*GRIDNY*GRIDNZ

#endif  // #define VOXELGRID_DEFINES



#include "real_defines.h"



#include "CUDA_HandleError.hpp"
#include "FileTalk.hpp"

#include "ChordsCalc_kernel3.cu"
#include "MeasurementSetup.hpp"
#include "VoxelGrid.hpp"
#include "CudaMS.hpp"
#include "CudaVG.hpp"
#include "CudaTransform.hpp"
#include "H5Reader.hpp"
#include "H5DensityWriter.hpp"
#include "visualization.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>



template<typename T, typename ConcreteVoxelGrid>
class WriteableCudaVG : public CudaVG<T, ConcreteVoxelGrid>
{
  public:

    WriteableCudaVG(
          T const   gridO0, T const   gridO1, T const   gridO2,
          T const   gridD0, T const   gridD1, T const   gridD2,
          int const gridN0, int const gridN1, int const gridN2 )
    : CudaVG<T, ConcreteVoxelGrid>(
          gridO0, gridO1, gridO2,
          gridD0, gridD1, gridD2,
          gridN0, gridN1, gridN2) {}

    void getOrigin( float * origin )
    {
      for(int dim=0; dim<3; dim++)
        origin[dim] = this->hostRepr()->gridO[dim];
    }

    void getVoxelSize( float * voxelSize )
    {
      for(int dim=0; dim<3; dim++)
        voxelSize[dim] = this->hostRepr()->gridD[dim];
    }

    void getNumberOfVoxels( int * numberOfVoxels )
    {
      for(int dim=0; dim<3; dim++)
        numberOfVoxels[dim] = this->hostRepr()->gridN[dim];
    }
};



template<typename T>
struct MeasurementEvent
{
  T   _value;
  int _channel;
  
  __host__ __device__
  MeasurementEvent()
  : _value(0.), _channel(-1) {}

  __host__ __device__
  MeasurementEvent( T value_, int channel_)
  : _value(value_), _channel(channel_) {}

  __host__ __device__
  MeasurementEvent( MeasurementEvent<T> const & ori )
  {
    _value   = ori._value;
    _channel = ori._channel;
  }
  
  __host__ __device__
  ~MeasurementEvent()
  {}

  __host__ __device__
  void operator=( MeasurementEvent<T> const & rhs )
  {
    _value   = rhs._value;
    _channel = rhs._channel;
  }

  __host__ __device__
  T value() const
  {
    return _value;
  }

  __host__ __device__
  int channel() const
  {
    return _channel;
  }
};



//#define CHUNKSIZE 400000
#define CHUNKSIZE 100                             // number of lines in one chunk
#define UPPERCHUNKID ((NCHANNELS+CHUNKSIZE-1)/CHUNKSIZE)
//#define UPPERCHUNKID 1
#define RANDOM_SEED 1234
#define NTHREADRAYS 100

typedef float val_t;

int main( int ac, char ** av )
{
  /* ---------------------------
   * Treat commandline arguments 
   * --------------------------- */
  SAYLINES(__LINE__-3, __LINE__-1);
  
  if(ac < 2)
  {
    std::cerr << "Wrong number of arguments. Exspected arguments:" << std::endl
              << "    1.: measurement filename (mandatory)" << std::endl
              << "    2.: file output prefix (optional, defaults to \"real_algo_output\")"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string fn(av[1]);
  std::string outpre;
  if(ac >= 3)
    outpre = std::string(av[2]);
  else
    outpre = std::string("test_angular_index_counting");
  

  /* --------------
   * Create objects
   * -------------- */
  SAYLINES(__LINE__-3, __LINE__-1);
  
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
  CudaMatrix<val_t,val_t>   chunk(CHUNKSIZE, VGRIDSIZE);
  
  for(int rowId=0; rowId<CHUNKSIZE; rowId++)
    for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
      chunk.set(rowId, vxlId, 0.);
  
  /* Measurement vector */
  //CudaVector<val_t, val_t>  yValues_chunk(CHUNKSIZE);

  CudaVector<MeasurementEvent<val_t>, MeasurementEvent<val_t> > 
                            y_chunk(CHUNKSIZE); // chunk part of meas.

  for(int listId=0; listId<CHUNKSIZE; listId++)
    y_chunk.set(listId, MeasurementEvent<val_t>(0., -1));
 
  
  /* ----------------
   * Read measurement
   * ---------------- */
  SAYLINES(__LINE__-3, __LINE__-1);
  
  std::cout << "Total number of channels:" << std::endl
            << "    " << NCHANNELS << std::endl;
  
  /* Allocate memory for and read raw input data */
  H5Reader h5reader(fn);
  val_t * meas = new val_t[NCHANNELS];
  h5reader.read(meas);
  
  /* Count those channels, that have values != 0. */
  int count(0);
  for(int cnlId=0; cnlId<NCHANNELS; cnlId++)
    if(meas[cnlId] != 0.)
      count++;

  int const NEVENTS(count);
  int const NCHUNKS((NEVENTS+CHUNKSIZE-1)/CHUNKSIZE);
  
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


  /* -----------------------
   * Calculate system matrix
   * ----------------------- */
  SAYLINES(__LINE__-3, __LINE__-1);

  /* Allocate memory for rays */
  val_t   rays_host[NCHUNKS*CHUNKSIZE*NTHREADRAYS*6*sizeof(val_t)];
  val_t * rays_devi;
  HANDLE_ERROR( cudaMalloc((void**)&rays_devi,
                NCHUNKS*CHUNKSIZE*NTHREADRAYS*6*sizeof(val_t)) );
  
  /* Iterate over chunks */
  for(int chunkId=0; (chunkId<UPPERCHUNKID) && (chunkId<NCHUNKS); chunkId++)
  {
    /* Copy chunk's part of measurement vector */
    SAYLINE(__LINE__-1);
    
    for(int listId=0; listId<CHUNKSIZE; listId++)
    {
      MeasurementEvent<val_t> event;
      if(chunkId*CHUNKSIZE + listId < NEVENTS)
        event = y.get(chunkId*CHUNKSIZE + listId);
      else
        event = MeasurementEvent<val_t>(0., -1);

      assert(!isnan(event.value()));
      assert(!isinf(event.value()));
      //std::cout << "channel " << event.channel() << ": " << event.value()
      //          << std::endl;

      y_chunk.set(      listId, event);
      //yValues_chunk.set(listId, event.value());
    }
    
#ifdef DEBUG 
/**//* Print measurement vector */
/**/SAYLINE(__LINE__-1);
/**/std::cout << std::left
/**/          << std::setw(16) << "y_chunk:"
/**/          << std::setw(16) << "yValues_chunk:"
/**/          << std::endl;
/**/for(int listId=0; listId<CHUNKSIZE; listId++)
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
    SAYLINE(__LINE__-1);
    for(int listId=0; listId<CHUNKSIZE; listId++)
      for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
        chunk.set(listId, vxlId, 0.);
    HANDLE_ERROR( cudaDeviceSynchronize() );

    /* Calculate system matrix chunk */
    SAYLINE(__LINE__-1);
    chordsCalc(
          chunkId, NCHANNELS, CHUNKSIZE, 1,
          static_cast<val_t*>(chunk.data()),
          rays_devi,
          &y_chunk,
          grid,
          VGRIDSIZE,
          setup,
          RANDOM_SEED,
          NTHREADRAYS);
    HANDLE_ERROR( cudaDeviceSynchronize() );
    chunk.set_devi_data_changed();
    
    for(int listId=0; listId<CHUNKSIZE; listId++)
    {
      for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
      {
        val_t elem = chunk.get(listId, vxlId);
        assert(!isnan(elem));
        assert(!isinf(elem));
      }
    }

#ifdef DEBUG
/**//* Print system matrix chunk */
/**/SAYLINE(__LINE__-1);
/**/std::cout << "chunk:" << std::endl;
/**/for(int listId=0; listId<CHUNKSIZE; listId++)
/**/{
/**/  int count(0);
/**/  for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
/**/    if(chunk.get(listId, vxlId) != 0.) count++;
/**/
/**/  if(count > 0)
/**/  {
/**/    std::cout << "  listId " << listId << ":  ";
/**/    for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
/**/    {
/**/      val_t elem = chunk.get(listId, vxlId);
/**/      if(elem != 0.)
/**/        std::cout << elem << "  ";
/**/    }
/**/    std::cout << std::endl;
/**/  }
/**/}
#endif  // DEBUG
  } /* End iterate over chunks */


  /* ----------------
   * File output
   * ---------------- */
  SAYLINES(__LINE__-3, __LINE__-1);
  
  /* Visualize grid */
  std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
  SAYLINE(__LINE__-1);
  DefaultVoxelGrid<val_t> * hostRepr = grid->hostRepr();
  SAYLINE(__LINE__-1);
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

  /* Visualize rays */
  HANDLE_ERROR( cudaMemcpy(rays_host, rays_devi,
                           6*NCHUNKS*CHUNKSIZE*NTHREADRAYS*sizeof(val_t),
                           cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaDeviceSynchronize() );

  for(int eventId=0; eventId<NEVENTS; eventId++)
  {
    BetaCompositePlyGeom compositeLines("");
    BetaPlyLine<val_t> lines[NEVENTS*NTHREADRAYS];
    for(int rayId=0; rayId<NTHREADRAYS; rayId++)
    {
      lines[rayId] = BetaPlyLine<val_t>("", &rays_host[6*(eventId*NTHREADRAYS + rayId)]);
      compositeLines.add(&lines[rayId]);
    }

    std::stringstream fn("");
    fn << outpre << "_rays-" << std::setw(3) << std::setfill('0') << eventId << ".ply";
    PlyWriter raysWriter(fn.str());
    raysWriter.write(compositeLines);
    raysWriter.close();
  }


  return 0;
}
