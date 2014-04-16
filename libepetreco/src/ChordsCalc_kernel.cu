#define DEBUG
#define PRINT_KERNEL 2

#include "CUDA_HandleError.hpp"
#include "ChordsCalc_lowlevel.hpp"
#include <math.h>
#include <curand_kernel.h>
#include <cstdio>
#include <assert.h>

//#define N0Z 13      // 1st detector's number of segments in z
//#define N0Y 13      // 1st detector's number of segments in y
//#define N1Z 13      // 2nd detector's number of segments in z
//#define N1Y 13      // 2nd detector's number of segments in y
//#define NA  180     // number of angular positions
//#define DA  2.      // angular step
//#define POS0X -45.7 // position of 1st detector's center in x
//#define POS1X  45.7 // position of 2nd detector's center in x
//#define SEGX 2.0    // x extent of one detector segment
//#define SEGY 0.4    // y extent of one detector segment
//#define SEGZ 0.4    // z extent of one detector segment

#define N0Z 5       // 1st detector's number of segments in z
#define N0Y 1       // 1st detector's number of segments in y
#define N1Z 4       // 2nd detector's number of segments in z
#define N1Y 1       // 2nd detector's number of segments in y
#define NA  1       // number of angular positions
#define DA  5.      // angular step
#define POS0X -3.5  // position of 1st detector's center in x
#define POS1X  3.5  // position of 2nd detector's center in x
#define SEGX 1.     // x edge length of one detector segment
#define SEGY 1.     // y edge length of one detector segment
#define SEGZ 1.     // z edge length of one detector segment

#define GRIDNX 3    // x dimension of voxel grid
#define GRIDNY 1    // y dimension of voxel grid
#define GRIDNZ 4    // z dimension od voxel grid
#define VGRIDSIZE GRIDNX*GRIDNY*GRIDNZ

typedef float val_t;

/**
 * @brief Get linearized index of voxel.
 * 
 * @param idx Index of voxel in x direction
 * @param idy Index of voxel in y direction
 * @param idz Index of voxel in z direction
 * @param gridN Axes' dimensions
 */
__inline__ __host__ __device__
int getLinVoxelId(
      int const idx, int const idy, int const idz,
      int const * const gridN)
{
  return  idz * gridN[0]*gridN[1]
        + idy * gridN[0]
        + idx;
}

/**
 * @brief Get linearized index of channel.
 *
 * @param ida Angular index
 * @param id0z Index of detector 0 segment in z direction
 * @param id0y Index of detector 0 segment in y direction
 * @param id1z Index of detector 1 segment in z direction
 * @param id1y Index of detector 1 segment in y direction
 */
__inline__
int getLinChannelId(
      int const ida,
      int const id0z, int const id0y,
      int const id1z, int const id1y )
{
  return   ida  * N0Z*N0Y*N1Z*N1Y
         + id0z *     N0Y*N1Z*N1Y
         + id0y *         N1Z*N1Y
         + id1z *             N1Y
         + id1y;
}

/**
 * @brief Get indices of channel configuration, seperately.
 *
 * @param 5DChannelId Result memory (int[5])
 * @param linChannelId Linear index of channel
 */
__host__ __device__
void get5DChannelId(
      int * const dimChannelId,
      int const linChannelId)
{
  int temp( linChannelId );
  dimChannelId[0] = temp / (N1Y*N1Z*N0Y*N0Z); // angular index
  temp %= (N1Y*N1Z*N0Y*N0Z);
  dimChannelId[1] = temp / (N1Y*N1Z*N0Y);     // det0z index
  temp %= (N1Y*N1Z*N0Y);
  dimChannelId[2] = temp / (N1Y*N1Z);         // det0y index
  temp %= (N1Y*N1Z);
  dimChannelId[3] = temp / (N1Y);             // det1z index
  temp %= (N1Y);
  dimChannelId[4] = temp;                     // det1y index
}

/**
 * @brief Get a channel's geometrical properties.
 *
 * @param pos0 Result memory (val_t[3]), position of detector0 segment's center
 * @param pos1 Result memory (val_t[3]), position of detector1 segment's center
 * @param edges Result memory (val_t[3]), lengths of detector segments' edges
 * @param sin_ Result memory (val_t *), sine of angle
 * @param cos_ Result memory (val_t *), cosine of angle
 * @param 5DChannelId Indices of channel configuration
 */
__host__ __device__
void getGeomProps(
      val_t * const pos0, val_t * const pos1,
      val_t * const edges,
      val_t * const sin_, val_t * const cos_,
      int const * const dimChannelId)
{
  pos0[0]  = POS0X;
  pos0[1]  = (dimChannelId[2]-0.5*N0Y+0.5)*SEGY;
  pos0[2]  = (dimChannelId[1]-0.5*N0Z+0.5)*SEGZ;
  pos1[0]  = POS1X;
  pos1[1]  = (dimChannelId[4]-0.5*N1Y+0.5)*SEGY;
  pos1[2]  = (dimChannelId[3]-0.5*N1Z+0.5)*SEGZ;
  edges[0] = SEGX;
  edges[1] = SEGY;
  edges[2] = SEGZ;
  sin_[0]  = sin(dimChannelId[0]*DA);
  cos_[0]  = cos(dimChannelId[0]*DA);
}

/**
 * @brief Get tranformation matrix (0..1, 0..1, 0..1) -> detector segment
 *        volume.
 * 
 * @param trafo Result memory (val_t[12])
 * @param pos Position of detector segment's center
 * @param edges Edge lengths of detector segment
 * @param sin Sine of channel angle
 * @param cos Cosine of channel angle
 */
__host__ __device__
void getTransformation(
      val_t * const trafo,
      val_t const * const pos, val_t const * const edges,
      val_t const sin, val_t const cos)
{
  trafo[0*4 + 0] = cos*edges[0];
  trafo[0*4 + 1] = 0.;
  trafo[0*4 + 2] = sin*edges[2];
  trafo[0*4 + 3] = cos*(pos[0]-.5*edges[0])\
                  +sin*(pos[2]-.5*edges[2]);
  trafo[1*4 + 0] = 0.;
  trafo[1*4 + 1] = edges[1];
  trafo[1*4 + 2] = 0.;
  trafo[1*4 + 3] = pos[1]-.5*edges[1];
  trafo[2*4 + 0] =-sin*edges[0];
  trafo[2*4 + 1] = 0.;
  trafo[2*4 + 2] = cos*edges[2];
  trafo[2*4 + 3] =-sin*(pos[0]-.5*edges[0])\
                  +cos*(pos[2]-.5*edges[2]);
}

/**
 * @brief Get random ray in channel.
 *
 * @aram ray Result memory (val_t[6]), start, end coordinates - in this order
 * @param linChannelId Linear index of channel
 */
#define SEED 1234
__device__
void getRay(
      val_t * const ray,
      int const linChannelId)
{
  int id = blockDim.x * blockIdx.x + threadIdx.x;

  // Get dimensional channel id
  int dimChannelId[5];
  get5DChannelId(dimChannelId, linChannelId);

  // Get geometrical properties of the channel
  val_t pos0[3];
  val_t pos1[3];
  val_t edges[3];
  val_t sin, cos;
  getGeomProps(pos0, pos1, edges, &sin, &cos, dimChannelId);
#ifdef DEBUG
  if(id == PRINT_KERNEL)
  {
    printf("getRay(...):\n");
    printf("    pos0 : %f, %f, %f\n", pos0[0],  pos0[1],  pos0[2]);
    printf("    pos1 : %f, %f, %f\n", pos1[0],  pos1[1],  pos1[2]);
    printf("    edges: %f, %f, %f\n", edges[0], edges[1], edges[2]);
    printf("    angle: %f\n", dimChannelId[0]*DA);
  }
#endif
  
  // Get transformation matrices
  val_t trafo0[12];
  getTransformation(trafo0, pos0, edges, sin, cos);
#ifdef DEBUG
  if(id == PRINT_KERNEL)
  {
    printf( "    -----\n");
    printf( "    trafo0:\n");
    printf( "    /  %05.2f  %05.2f  %05.2f  %05.2f  \\\n",
           trafo0[0], trafo0[1], trafo0[2],  trafo0[3]);
    printf( "    |  %05.2f  %05.2f  %05.2f  %05.2f  |\n",
           trafo0[4], trafo0[5], trafo0[6],  trafo0[7]);
    printf("    \\  %05.2f  %05.2f  %05.2f  %05.2f  /\n",
           trafo0[8], trafo0[9], trafo0[10], trafo0[11]);
  }
#endif

  val_t trafo1[12];
  getTransformation(trafo1, pos1, edges, sin, cos);
#ifdef DEBUG
  if(id == PRINT_KERNEL)
  {
    printf( "    -----\n");
    printf( "    trafo1:\n");
    printf( "    /  %05.2f  %05.2f  %05.2f  %05.2f  \\\n",
           trafo1[0], trafo1[1], trafo1[2],  trafo1[3]);
    printf( "    |  %05.2f  %05.2f  %05.2f  %05.2f  |\n",
           trafo1[4], trafo1[5], trafo1[6],  trafo1[7]);
    printf("    \\  %05.2f  %05.2f  %05.2f  %05.2f  /\n",
           trafo1[8], trafo1[9], trafo1[10], trafo1[11]);
  }
#endif

  // Get homogenuous seed coordinates for ray start, end
  curandState state;
  curand_init(SEED, id, 0, &state);
  val_t rand[8];
  for(int i=0; i<3; i++)
  {
    rand[i]   = curand_uniform(&state);
    rand[i+4] = curand_uniform(&state);
  }
  rand[3] = 1.;
  rand[7] = 1.;
#ifdef DEBUG
  if(id == PRINT_KERNEL)
  {
    printf( "    -----\n");
    printf("    rand: %05.3f,  %05.3f,  %05.3f,  %05.3f,\n",
            rand[0], rand[1], rand[2], rand[3]); 
    printf("          %05.3f,  %05.3f,  %05.3f,  %05.3f,\n",
            rand[4], rand[5], rand[6], rand[7]);
  }
#endif
  
  // Transform to obtain start, end
  for(int i=0; i<3; i++)
  {
    ray[i]   = 0.;
    ray[i+3] = 0.;

    for(int j=0; j<4; j++)
    {
      ray[i]   += trafo0[i*4 + j] * rand[j];
      ray[i+3] += trafo1[i*4 + j] * rand[j+4];
    }
  }
}

/**
 * @brief Kernel function for calculation of chords (=intersection line of ray
 *        with one voxel)
 *
 * @param chords Result memory
 * @param linearVoxelId Result memory, linear voxel index
 * @param rays Result memory
 * @param linChannelId Linear index of channel that is calculated
 * @param gridO Grid origin
 * @param gridD Grid voxels edge lengths
 * @param gridN Grid dimensions
 */
__global__
void chordsCalc(
      val_t * const chords, int * const linearVoxelId,
      val_t * const rays,
      int const linearChannelId,
      val_t const * gridO, val_t const * const gridD, int const * const gridN )
{
  int globalId = blockDim.x * blockIdx.x + threadIdx.x;

  // Get ray
  val_t ray[6];
  getRay(ray, linearChannelId);
#ifdef DEBUG
  if(globalId == PRINT_KERNEL)
  {
    printf("start:  %06.3f,  %06.3f,  %06.3f\n",
            ray[0], ray[1], ray[2]);
    printf("end  :  %06.3f,  %06.3f,  %06.3f\n",
            ray[3], ray[4], ray[5]);
  }
#endif
  
  for(int comp=0; comp<6; comp++)
    rays[6*globalId + comp] = ray[comp];

  // ##################
  // ### INITIALIZATION
  // ##################

  // Get intersection minima for all axes, get intersection info
  val_t aDimmin[3];
  val_t aDimmax[3];
  bool  crosses[3];
#ifdef DEBUG
  if(globalId == PRINT_KERNEL)
  {
    for(int dim=0; dim<3; dim++)
    {
      printf("alpha[%i](%i): %f\n", dim, 0,
              alphaFromId(0,          dim, ray, gridO, gridD, gridN));

      printf("alpha[%i](%i): %f\n", dim, gridN[dim],
              alphaFromId(gridN[dim], dim, ray, gridO, gridD, gridN));
    }
  }
#endif
 
  getAlphaDimmin(   aDimmin, ray, gridO, gridD, gridN);
  getAlphaDimmax(   aDimmax, ray, gridO, gridD, gridN);
  getCrossesPlanes( crosses, ray, gridO, gridD, gridN);
#ifdef DEBUG
  if(globalId == PRINT_KERNEL)
  {
    printf("aDimmin:  %6f,  %6f,  %6f\n",
            aDimmin[0], aDimmin[1], aDimmin[2] );
    printf("aDimmax:  %6f,  %6f,  %6f\n",
            aDimmax[0], aDimmax[1], aDimmax[2] );
    printf("crosses:  %i,  %i,  %i\n",
            crosses[0], crosses[1], crosses[2] );
  }
#endif
  
  // Get parameter of the entry and exit points
  val_t aMin;
  val_t aMax;
  bool  aMinGood;
  bool  aMaxGood;
  getAlphaMin(  &aMin, &aMinGood, aDimmin, crosses);
  getAlphaMax(  &aMax, &aMaxGood, aDimmax, crosses);
  // Do entry and exit points lie in beween ray start and end points?
  aMinGood &= (aMin >= 0. && aMin <= 1.);
  aMaxGood &= (aMax >= 0. && aMax <= 1.);
  // Is grid intersected at all, does ray start and end outside the grid?
  // - otherwise return
  //if(aMin>aMax || !aMinGood || !aMaxGood) return;
  if(aMin>aMax || !aMinGood || !aMaxGood)
  {
    //printf("fail\n");
    return;
  }

  // Get length of ray
  val_t const length(getLength(ray));
  
  // Get parameter update values 
  val_t aDimup[3];
  getAlphaDimup(  aDimup, ray, gridD);
  
  // Get id update values
  int idDimup[3];
  getIdDimup( idDimup, ray);
  
  // Initialize array of next parameters
  val_t aDimnext[3];
  for(int dim=0; dim<3; dim++) aDimnext[dim] = aDimmin[dim] + aDimup[dim];
  
  // Initialize array of voxel indices
  int id[3];
  val_t aNext;
  bool aNextExists;
  MinFunctor<3>()(&aNext, &aNextExists, aDimnext, crosses);

  for(int dim=0; dim<3; dim++)
    id[dim] = floor(phiFromAlpha(
          float(0.5)*(aMin + aNext), dim, ray, gridO, gridD, gridN
                                     )
                        );

  // Initialize current parameter
  val_t aCurr = aMin;
  
  
  // ##################
  // ###  ITERATIONS
  // ##################
  int chordId = 0;
  while(aCurr < aMax)
  {
    assert(chordId<GRIDNX*GRIDNY*GRIDNZ);

    // Get parameter of next intersection
    MinFunctor<3>()(&aNext, &aNextExists, aDimnext, crosses);
    
    bool anyAxisCrossed = false; 
    // For all axes...
    for(int dim=0; dim<3; dim++)
    {
      // Is this axis' plane crossed at the next parameter of intersection?
      bool dimCrossed = (aDimnext[dim] == aNext);
      anyAxisCrossed |= dimCrossed;
      

      // If this axis' plane is crossed ...
      //      ... clear and write chord length and voxel index
      //chords[     chordId]
      //        *= (int)(!dimCrossed);
      //chords[     chordId]
      //        += (int)( dimCrossed) * (aDimnext[dim]-aCurr)*length;
      atomicAdd(&chords[getLinVoxelId(id[0], id[1], id[2], gridN)],
                (int)(dimCrossed) * (aDimnext[dim]-aCurr)*length);
      //linearVoxelId[chordId]
      //        *= (int)(!dimCrossed);
      //linearVoxelId[chordId]
      //        += (int)( dimCrossed) * getLinVoxelId(id[0], id[1], id[2], gridN);
      
      //      ... increase chord index (writing index)
      chordId       +=  (int)(dimCrossed);
      
      //      ... update current parameter
      aCurr          = (int)(!dimCrossed) * aCurr
                      + (int)(dimCrossed) * aDimnext[dim];
      //      ... update this axis' paramter to next plane
      aDimnext[dim] +=  (int)(dimCrossed) * aDimup[dim];
      //      ... update this axis' voxel index
      id[dim]       +=  (int)(dimCrossed) * idDimup[dim];

    }
  }
}




#include "Ply.hpp"
#include "TemplateVertex.hpp"
typedef TemplateVertex<val_t> Vertex;

class CPlyGrid : public PlyGrid<Vertex>
{
  public:
    
    enum OriginCMode
    {
      AT_ORIGIN,
    };

    enum CenterCMode
    {
      AT_CENTER,
    };

    CPlyGrid(
          std::string const name,
          val_t const * const gridO,
          val_t const * const gridD,
          int const * const gridN )
    : PlyGrid<Vertex>(name,
                      Vertex(gridO[0], gridO[1], gridO[2]),
                      gridN[0]+1, gridN[1]+1, gridN[2]+1,
                      gridD[0], gridD[1], gridD[2]) {}
    
    CPlyGrid(
          std::string const name,
          val_t const * const gridAt,
          val_t const * const gridD,
          int const * const gridN,
          OriginCMode cmode )
    : PlyGrid<Vertex>(name,
                      Vertex(gridAt[0], gridAt[1], gridAt[2]),
                      gridN[0]+1, gridN[1]+1, gridN[2]+1,
                      gridD[0], gridD[1], gridD[2]) {}
    
    CPlyGrid(
          std::string const name,
          val_t const * const gridAt,
          val_t const * const gridD,
          int const * const gridN,
          CenterCMode cmode )
    : PlyGrid<Vertex>(name,
                      Vertex(gridAt[0]-0.5*gridN[0]*gridD[0],
                             gridAt[1]-0.5*gridN[1]*gridD[1],
                             gridAt[2]-0.5*gridN[2]*gridD[2]),
                      gridN[0]+1, gridN[1]+1, gridN[2]+1,
                      gridD[0], gridD[1], gridD[2]) {}
};

class CCompositePlyGeom : public CompositePlyGeometry
{
  public:
    
    CCompositePlyGeom( std::string const name )
    : CompositePlyGeometry(name) {}
};

class CPlyLine : public PlyLine<Vertex>
{
  public:
    
    CPlyLine()
    : PlyLine("", Vertex(0,0,0), Vertex(0,0,0)) {}

    CPlyLine(
          std::string const name,
          val_t const * const ray)
    : PlyLine(name,
              Vertex(ray[0], ray[1], ray[2]),
              Vertex(ray[3], ray[4], ray[5])) {}
};


#define NBLOCKS 1
#define NTHREADS 1024
int main()
{
  // Host memory allocation, initialization
  int   gridN_host[] = {GRIDNX, GRIDNY, GRIDNZ};
  val_t gridO_host[] = {-1.5, -0.5, -2.0};
  val_t gridD_host[] = {1.0, 1.0, 1.0};

  int linearChannelId = 1;

  val_t chords_host  [VGRIDSIZE];
  for(int i=0; i<VGRIDSIZE; i++)
    chords_host[i]=0;
  int   voxelIds_host[VGRIDSIZE];

  val_t rays_host[6*NBLOCKS*NTHREADS];
  
  // Visualize grid
  PlyGrid<Vertex> grid("", Vertex(gridO_host[0], gridO_host[1], gridO_host[2]),
                       gridN_host[0]+1, gridN_host[1]+1, gridN_host[2]+1,
                       gridD_host[0], gridD_host[1], gridD_host[2]);
  PlyWriter writer("ChordsCalc_kernel2_grid.ply");
  writer.write(grid);
  writer.close();
  
  // Visualize det0
  int   det0N[] = {1, N0Y, N0Z};
  val_t det0C[] = {POS0X, 0, 0};
  val_t detD[]  = {SEGX, SEGY, SEGZ};
  CPlyGrid det0("", det0C, detD, det0N, CPlyGrid::AT_CENTER);
  PlyWriter det0Writer("ChordsCalc_kernel2_det0.ply");
  det0Writer.write(det0);
  det0Writer.close();

  // Visualize det1
  int   det1N[] = {1, N1Y, N1Z};
  val_t det1C[] = {POS1X, 0, 0};
  CPlyGrid det1("", det1C, detD, det1N, CPlyGrid::AT_CENTER);
  PlyWriter det1Writer("ChordsCalc_kernel2_det1.ply");
  det1Writer.write(det1);
  det1Writer.close();
  
  // Device memory allocation
  val_t * chords_devi;
  HANDLE_ERROR( cudaMalloc((void**)&chords_devi, VGRIDSIZE*sizeof(val_t)) );
  int *   voxelIds_devi;
  HANDLE_ERROR( cudaMalloc((void**)&voxelIds_devi, VGRIDSIZE*sizeof(int)) );
  val_t * rays_devi;
  HANDLE_ERROR( cudaMalloc((void**)&rays_devi, 6*NBLOCKS*NTHREADS*sizeof(val_t)) );
  int *   gridN_devi;
  HANDLE_ERROR( cudaMalloc((void**)&gridN_devi, 3*sizeof(int)) );
  val_t * gridO_devi;
  HANDLE_ERROR( cudaMalloc((void**)&gridO_devi, 3*sizeof(val_t)) );
  val_t * gridD_devi;
  HANDLE_ERROR( cudaMalloc((void**)&gridD_devi, 3*sizeof(val_t)) );
  
  // Copy host to device
  HANDLE_ERROR( cudaMemcpy(chords_devi, chords_host,
                           VGRIDSIZE*sizeof(val_t),
                           cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(gridN_devi, gridN_host,
                           3*sizeof(int),
                           cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(gridO_devi, gridO_host,
                           3*sizeof(val_t),
                           cudaMemcpyHostToDevice) );
  HANDLE_ERROR( cudaMemcpy(gridD_devi, gridD_host,
                           3*sizeof(val_t),
                           cudaMemcpyHostToDevice) );
  
  // Run kernel
  chordsCalc<<<NBLOCKS,NTHREADS>>>(chords_devi, voxelIds_devi, rays_devi,
                                   linearChannelId,
                                   gridO_devi, gridD_devi, gridN_devi );
  HANDLE_ERROR( cudaDeviceSynchronize() );
  
  // Copy results device to host
  HANDLE_ERROR( cudaMemcpy(chords_host, chords_devi,
                           VGRIDSIZE*sizeof(val_t),
                           cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaMemcpy(voxelIds_host, voxelIds_devi,
                           VGRIDSIZE*sizeof(int),
                           cudaMemcpyDeviceToHost) );
  HANDLE_ERROR( cudaMemcpy(rays_host, rays_devi,
                           6*NBLOCKS*NTHREADS*sizeof(val_t),
                           cudaMemcpyDeviceToHost) );
  for(int i=0; i<VGRIDSIZE; i++)
  {
    std::cout << "voxel " << i << ": " << chords_host[i]/NTHREADS << std::endl;
  }

  // Visualize rays
  CCompositePlyGeom compositeLines("");
  CPlyLine lines[NBLOCKS*NTHREADS];
  for(int i=0; i<NBLOCKS*NTHREADS; i++)
  {
    lines[i] = CPlyLine("", &rays_host[6*i]);
    compositeLines.add(&lines[i]);
  }
  PlyWriter raysWriter("ChordsCalc_kernel2_rays.ply");
  raysWriter.write(compositeLines);
  raysWriter.close();
  
  // Clean up
  cudaFree(chords_devi);
  cudaFree(voxelIds_devi);
  cudaFree(gridN_devi);
  cudaFree(gridO_devi);
  cudaFree(gridD_devi);

  return 0;
}
