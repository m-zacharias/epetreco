#define N0Z 13        // 1st detector's number of segments in z
#define N0Y 13        // 1st detector's number of segments in y
#define N1Z 13        // 2nd detector's number of segments in z
#define N1Y 13        // 2nd detector's number of segments in y
#define NA  180       // number of angular positions
#define DA  2.        // angular step
#define POS0X -0.457  // position of 1st detector's center in x [m]
#define POS1X  0.457  // position of 2nd detector's center in x [m]
#define SEGX 0.02     // x edge length of one detector segment [m]
#define SEGY 0.004    // y edge length of one detector segment [m]
#define SEGZ 0.004    // z edge length of one detector segment [m]
#define NCHANNELS NA*N0Z*N0Y*N1Z*N1Y

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

#define RANDOM_SEED 1234
#define NTHREADRAYS 100

#include "CUDA_HandleError.hpp"
#include "FileTalk.hpp"

//#include "ChordsCalc_kernelWrapper.hpp"
#include "ChordsCalc_kernel2.cu"
#include "MeasurementSetup.hpp"
#include "VoxelGrid.hpp"
#include "CudaMS.hpp"
#include "CudaVG.hpp"
#include "CudaTransform.hpp"
#include "H5Reader.hpp"
#include "H5DensityWriter.hpp"
#include "visualization.hpp"



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


template<typename TE, typename TI>
class MyCudaVector : public CudaVector<TE,TI>
{
  public:
    
    MyCudaVector( int n )
    : CudaVector<TE,TI>(n) {}

    void * hostData()
    {
      return CudaVector<TE,TI>::_raw_host;
    }

    void set_host_data_changed()
    {
      CudaVector<TE,TI>::_host_data_changed = true;
    }
};



//#define CHUNKSIZE 400000
#define CHUNKSIZE 1000                             // number of lines in one chunk
#define NCHUNKS (NCHANNELS+CHUNKSIZE-1)/CHUNKSIZE

typedef float val_t;

int main( int ac, char ** av )
{
  /* Start */
  std::cout << "Test" << std::endl << std::flush;
  if(ac != 2)
  {
    std::cerr << "Wrong number of arguments. Exspected: 1: measurement_filename"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  std::string fn(av[1]);
  
  WriteableCudaVG<val_t, DefaultVoxelGrid<val_t> > * grid =
        new WriteableCudaVG<val_t, DefaultVoxelGrid<val_t> >(
              GRIDOX, GRIDOY, GRIDOZ,
              GRIDDX, GRIDDY, GRIDDZ,
              GRIDNX, GRIDNY, GRIDNZ);

//  DefaultVoxelGrid<val_t> * hostRepr = grid->hostRepr();
  
  CudaMS<val_t, DefaultMeasurementSetup<val_t> > * setup =
        new CudaMS<val_t, DefaultMeasurementSetup<val_t> >(
                POS0X, POS1X, NA, N0Z, N0Y, N1Z, N1Y,
                DA, SEGX, SEGY, SEGZ);
  
  CudaTransform<val_t,val_t>  trafo;
  CudaMatrix<val_t,val_t>     chunk(CHUNKSIZE, VGRIDSIZE);  // s m chunk
  MyCudaVector<val_t,val_t>   y(NCHANNELS);       // measurement
  MyCudaVector<val_t,val_t>   y_chunk(CHUNKSIZE); // chunk part of meas.
  CudaVector<val_t,val_t>     x(VGRIDSIZE);       // density guess
  for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
    x.set(voxelId, 1.);
  val_t one(1.);
  val_t zero(0.);
  
  /* ----------------
   * Read measurement
   * ---------------- */
  std::cout << NCHANNELS << std::endl;
  H5Reader h5reader(fn);
  val_t * meas = new val_t[NCHANNELS];
  h5reader.read(meas);
  for(int cnlId=0; cnlId<NCHANNELS; cnlId++)
    y.set(cnlId, meas[cnlId]);
  //for(int cnlId=0; cnlId<NCHANNELS; cnlId++)
  //  std::cout << y.get(cnlId) << "  ";
  //std::cout << std::endl;

  /* ----------
   * Iterations
   * ---------- */
  SAYLINES(__LINE__-3, __LINE__-1);
#define NITERATIONS 1
  for(int iteration=0; iteration<NITERATIONS; iteration++)  // for iterations
  {
    /* Set guess elements to null */
    SAYLINE(__LINE__-1);
    for(int voxelId=0; voxelId<VGRIDSIZE; voxelId++)
      x.set(voxelId, 0.);

    //for(int chunkId=0; chunkId<NCHUNKS; chunkId++)          // for chunks
    for(int chunkId=0; chunkId<1; chunkId++)          // for chunks
    {
      /* Copy chunk's part of measurement vector */
      SAYLINE(__LINE__-1);
      for(int channelId=0; channelId<CHUNKSIZE; channelId++)
        if(chunkId*CHUNKSIZE+channelId < NCHANNELS)
          y_chunk.set(channelId, y.get(chunkId*CHUNKSIZE+channelId));
        else
          y_chunk.set(channelId, 0.);
      for(int cnlId=0; cnlId<CHUNKSIZE; cnlId++)
        assert((!isnan(y_chunk.get(cnlId))) && (!isinf(y_chunk.get(cnlId))));
      //for(int cnlId=0; cnlId<CHUNKSIZE; cnlId++)
      //  std::cout << y_chunk.get(cnlId) << " ";
      //std::cout << std::endl;
      
      /* Set system matrix chunk's elements to null */
      SAYLINE(__LINE__-1);
      for(int cnlId=0; cnlId<CHUNKSIZE; cnlId++)
        for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
          chunk.set(cnlId, vxlId, 0.);
      HANDLE_ERROR( cudaDeviceSynchronize() );

      /* Calculate system matrix chunk */
      SAYLINE(__LINE__-1);
      chordsCalc_noVis(chunkId, NCHANNELS, CHUNKSIZE, 1,
                 static_cast<val_t*>(chunk.data()),
                 grid,
                 VGRIDSIZE,
                 setup);
      HANDLE_ERROR( cudaDeviceSynchronize() );
      chunk.set_devi_data_changed();
      for(int cnlId=0; cnlId<CHUNKSIZE; cnlId++)
      {
        for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
        {
          val_t elem = chunk.get(cnlId, vxlId);
          assert(!isnan(elem));
          assert(!isinf(elem));
          std::cout << elem << " ";
        }
      }
      std::cout << std::endl;

      /* Back projection */
      SAYLINE(__LINE__-1);
      trafo.gemv(BLAS_OP_T,
                 &one, &chunk,
                 &y_chunk,
                 &one, &x);
      x.set_devi_data_changed();
    } // for chunks
  } // for iterations

  ///* Prepare guess data */
  //for(int idx=0; idx<GRIDNX; idx++)
  //  for(int idy=0; idy<GRIDNY; idy++)
  //    for(int idz=0; idz<GRIDNZ; idz++)
  //    {
  //      int memid = getLinVoxelId(idx,idy,idz,grid->hostRepr()->gridN);
  //      if(idz==0)
  //        x.set(memid, 1.);
  //      else
  //        x.set(memid, 0.);
  //    }
  
  for(int vxlId=0; vxlId<VGRIDSIZE; vxlId++)
    std::cout << x.get(vxlId) << " ";
  std::cout << std::endl;

  /* Write last guess */
  SAYLINE(__LINE__-1);
  val_t * guess = new val_t[VGRIDSIZE];
  for(int memid=0; memid<VGRIDSIZE; memid++)
    guess[memid] = x.get(memid);

  H5DensityWriter<WriteableCudaVG<val_t, DefaultVoxelGrid<val_t> > > h5writer(std::string("x.h5"));
  h5writer.write(guess, *grid);
  
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
  PlyWriter writer("algo_grid.ply");
  writer.write(visGrid);
  writer.close();

  /* Visualize det0 */
  int   det0N[] = {1, N0Y, N0Z};
  val_t det0C[] = {POS0X, 0, 0};
  val_t detD[]  = {SEGX, SEGY, SEGZ};
  BetaPlyGrid<val_t> det0(
        "", det0C, detD, det0N, BetaPlyGrid<val_t>::AT_CENTER);
  PlyWriter det0Writer("ChordsCalc_kernel2_det0.ply");
  det0Writer.write(det0);
  det0Writer.close();

  /* Visualize det1 */
  int   det1N[] = {1, N1Y, N1Z};
  val_t det1C[] = {POS1X, 0, 0};
  BetaPlyGrid<val_t> det1("", det1C, detD, det1N, BetaPlyGrid<val_t>::AT_CENTER);
  PlyWriter det1Writer("ChordsCalc_kernel2_det1.ply");
  det1Writer.write(det1);
  det1Writer.close();

  return 0;
}
