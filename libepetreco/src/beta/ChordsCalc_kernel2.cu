#ifndef CHORDSCALC_KERNEL
#define CHORDSCALC_KERNEL

#include "ChordsCalc_lowlevel.hpp"
#include "defines.h"

#include <math.h>
#include <curand_kernel.h>
#include <cstdio>
#include <assert.h>

#ifndef PRINT_KERNEL
#define PRINT_KERNEL 0
#endif

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



__inline__ __host__ __device__
int getLinMtxId(
      int const rowId, int const rowDim,
      int const colId, int const colDim )
{
  return colId * colDim + rowId;
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
__device__
void getRay(
      val_t * const ray,
      curandState & state,
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
/**/if(id == PRINT_KERNEL)
/**/{
/**/  printf("getRay(...):\n");
/**/  printf("    pos0 : %f, %f, %f\n", pos0[0],  pos0[1],  pos0[2]);
/**/  printf("    pos1 : %f, %f, %f\n", pos1[0],  pos1[1],  pos1[2]);
/**/  printf("    edges: %f, %f, %f\n", edges[0], edges[1], edges[2]);
/**/  printf("    angle: %f\n", dimChannelId[0]*DA);
/**/}
#endif
  
  // Get transformation matrices
  val_t trafo0[12];
  getTransformation(trafo0, pos0, edges, sin, cos);
#ifdef DEBUG
/**/if(id == PRINT_KERNEL)
/**/{
/**/  printf( "    -----\n");
/**/  printf( "    trafo0:\n");
/**/  printf( "    /  %05.2f  %05.2f  %05.2f  %05.2f  \\\n",
/**/         trafo0[0], trafo0[1], trafo0[2],  trafo0[3]);
/**/  printf( "    |  %05.2f  %05.2f  %05.2f  %05.2f  |\n",
/**/         trafo0[4], trafo0[5], trafo0[6],  trafo0[7]);
/**/  printf("    \\  %05.2f  %05.2f  %05.2f  %05.2f  /\n",
/**/         trafo0[8], trafo0[9], trafo0[10], trafo0[11]);
/**/}
#endif

  val_t trafo1[12];
  getTransformation(trafo1, pos1, edges, sin, cos);
#ifdef DEBUG
/**/if(id == PRINT_KERNEL)
/**/{
/**/  printf( "    -----\n");
/**/  printf( "    trafo1:\n");
/**/  printf( "    /  %05.2f  %05.2f  %05.2f  %05.2f  \\\n",
/**/         trafo1[0], trafo1[1], trafo1[2],  trafo1[3]);
/**/  printf( "    |  %05.2f  %05.2f  %05.2f  %05.2f  |\n",
/**/         trafo1[4], trafo1[5], trafo1[6],  trafo1[7]);
/**/  printf("    \\  %05.2f  %05.2f  %05.2f  %05.2f  /\n",
/**/         trafo1[8], trafo1[9], trafo1[10], trafo1[11]);
/**/}
#endif

  // Get homogenuous seed coordinates for ray start, end
  //curandState state;
  //curand_init(SEED, id, 0, &state);
  val_t rand[8];
  for(int i=0; i<3; i++)
  {
    rand[i]   = curand_uniform(&state);
    rand[i+4] = curand_uniform(&state);
  }
  rand[3] = 1.;
  rand[7] = 1.;
#ifdef DEBUG
/**/if(id == PRINT_KERNEL)
/**/{
/**/  printf( "    -----\n");
/**/  printf("    rand: %05.3f,  %05.3f,  %05.3f,  %05.3f,\n",
/**/          rand[0], rand[1], rand[2], rand[3]); 
/**/  printf("          %05.3f,  %05.3f,  %05.3f,  %05.3f,\n",
/**/          rand[4], rand[5], rand[6], rand[7]);
/**/}
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
      val_t * const chords,
      val_t * const rays,
      val_t const * gridO, val_t const * const gridD, int const * const gridN,
      int const channelOffset, int const nChannels, int const chunkSize,
      int const vgridSize )
{
  int const globalId(blockDim.x * blockIdx.x + threadIdx.x);
#ifdef DEBUG
    if(globalId == PRINT_KERNEL)
    {
      printf("\nchordsCalc(...):\n");
    }
#endif
  // Global id of channel
  int const linChannelId(channelOffset + blockIdx.x);
  if(linChannelId >= nChannels)
    return;

  val_t ray[6];
  curandState kernelRandState;
  curand_init(RANDOM_SEED, linChannelId, 0, &kernelRandState);

  for(int iRay=0; iRay<NTHREADRAYS; iRay++)
  {
    // Get ray
    getRay(ray, kernelRandState, linChannelId);
#ifdef DEBUG
/**/if(globalId == PRINT_KERNEL)
/**/{
/**/  printf("\n");
/**/  printf("start:  %06.3f,  %06.3f,  %06.3f\n",
/**/          ray[0], ray[1], ray[2]);
/**/  printf("end  :  %06.3f,  %06.3f,  %06.3f\n",
/**/          ray[3], ray[4], ray[5]);
/**/}
#endif
    
    // Write ray 
    for(int dim=0; dim<6; dim++)
      rays[6*(globalId*NTHREADRAYS + iRay) + dim] = ray[dim];

    // ##################
    // ### INITIALIZATION
    // ##################
#ifdef DEBUG
/**/if(globalId == PRINT_KERNEL)
/**/{
/**/  printf("\n");
/**/  printf("### INITIALIZATION\n");
/**/}
#endif

    // Get intersection minima for all axes, get intersection info
    val_t aDimmin[3];
    val_t aDimmax[3];
    bool  crosses[3];
//#ifdef DEBUG
///**/if(globalId == PRINT_KERNEL)
///**/{
///**/  for(int dim=0; dim<3; dim++)
///**/  {
///**/    printf("alpha[%i](%i): %f\n", dim, 0,
///**/            alphaFromId(0,          dim, ray, gridO, gridD, gridN));
///**/
///**/    printf("alpha[%i](%i): %f\n", dim, gridN[dim],
///**/            alphaFromId(gridN[dim], dim, ray, gridO, gridD, gridN));
///**/  }
///**/}
//#endif
    getAlphaDimmin(   aDimmin, ray, gridO, gridD, gridN);
    getAlphaDimmax(   aDimmax, ray, gridO, gridD, gridN);
    getCrossesPlanes( crosses, ray, gridO, gridD, gridN);
//#ifdef DEBUG
///**/if(globalId == PRINT_KERNEL)
///**/{
///**/  printf("aDimmin:  %6f,  %6f,  %6f\n",
///**/          aDimmin[0], aDimmin[1], aDimmin[2] );
///**/  printf("aDimmax:  %6f,  %6f,  %6f\n",
///**/          aDimmax[0], aDimmax[1], aDimmax[2] );
///**/  printf("crosses:  %i,  %i,  %i\n",
///**/          crosses[0], crosses[1], crosses[2] );
///**/}
//#endif
  
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
    for(int dim=0; dim<3; dim++)
    {
      aDimnext[dim] = aDimmin[dim];
      while(aDimnext[dim]<=aMin)
        aDimnext[dim] += aDimup[dim];
    }

    // Initialize array of voxel indices
    int id[3];
    val_t aNext;
    bool aNextExists;
    MinFunctor<3>()(&aNext, &aNextExists, aDimnext, crosses);

#ifdef DEBUG
/**/  if(globalId == PRINT_KERNEL)
/**/  {
/**/    printf("aMin:        %f\n", aMin);
/**/    printf("aNext:       %f\n", aNext);
/**/    printf("aNextExists: %i\n", aNextExists);
/**/  }
#endif
    for(int dim=0; dim<3; dim++)
    {
      id[dim] = floor(phiFromAlpha(
            float(0.5)*(aMin + aNext), dim, ray, gridO, gridD, gridN
                                       )
                          );
#ifdef DEBUG
/**/  if(globalId == PRINT_KERNEL)
/**/  {
/**/    printf("phiFromAlpha: %f  ",
/**/           phiFromAlpha(float(0.5)*(aMin + aNext), dim, ray, gridO, gridD, gridN));
/**/    printf("id[%i]: %02i  ",
/**/           dim, id[dim]);
/**/  }
#endif
    } // for(dim)
#ifdef DEBUG
/**/  if(globalId == PRINT_KERNEL)
/**/  {
/**/    printf("\n");
/**/  }
#endif


    // Initialize current parameter
    val_t aCurr = aMin;

#ifdef DEBUG
/**/if(globalId == PRINT_KERNEL)
/**/{
/**/  printf("aMin:    %05.3f\n", aMin);
///**/  printf("aMax:    %05.3f\n", aMax);
/**/  printf("aMax:    %f\n", aMax);
/**/  printf("aDimup:  %05.3f,  %05.3f,  %05.3f\n",
             aDimup[0], aDimup[1], aDimup[2]);
/**/  printf("idDimup: %02i,  %02i,  %02i\n",
             idDimup[0], idDimup[1], idDimup[2]);
/**/  printf("length:  %05.3f\n", length);
/**/}
#endif
    
    
    // ##################
    // ###  ITERATIONS
    // ##################
#ifdef DEBUG
/**/if(globalId == PRINT_KERNEL)
/**/{
/**/  printf("\n");
/**/  printf("### ITERATIONS\n");
/**/}
#endif
    int chordId = 0;
//    while(aCurr < aMax)
    while(   id[0]<gridN[0]
          && id[1]<gridN[1]
          && id[2]<gridN[2]
          && id[0]>=0
          && id[1]>=0
          && id[2]>=0)
    {
#ifdef DEBUG
/**/if(globalId == PRINT_KERNEL)
/**/{
///**/  printf("aCurr: %05.2f\n", aCurr);
/**/  printf("aCurr: %f\n", aCurr);
/**/  printf("aCurr-aMax: %e\n", aCurr-aMax);
/**/  printf("aCurr<aMax: %i\n", aCurr<aMax);
/**/}
#endif
      assert(chordId<VGRIDSIZE);

      // Get parameter of next intersection
      MinFunctor<3>()(&aNext, &aNextExists, aDimnext, crosses);
      
      bool anyAxisCrossed = false; 
      // For all axes...
      for(int dim=0; dim<3; dim++)
      {
        // Is this axis' plane crossed at the next parameter of intersection?
        //bool dimCrossed = (aDimnext[dim] == aNext);
        bool dimCrossed = (aDimnext[dim] == aNext && aNextExists);
        anyAxisCrossed |= dimCrossed;

        // If this axis' plane is crossed ...
        //      ... write chord length at voxel index
#ifdef DEBUG
/**/    syncthreads();
/**/    if(globalId == PRINT_KERNEL)
/**/    {
/**/      printf("kernel: %i, a: %5.3f\n",
/**/             globalId,
/**/             (int)(dimCrossed) * (aDimnext[dim]-aCurr)*length);
/**/    }
/**/    syncthreads();
#endif
        assert(((int)(dimCrossed) * (aDimnext[dim]-aCurr)*length) >= 0);
        
        int rowId  = linChannelId % chunkSize;
        int rowDim = vgridSize;
        int colId  = getLinVoxelId(id[0], id[1], id[2], gridN);
        int colDim = chunkSize;
        int linMtxId = getLinMtxId(rowId, rowDim, colId, colDim);
        atomicAdd(&chords[linMtxId],
                  (int)(dimCrossed) * (aDimnext[dim]-aCurr)*length);

        //atomicAdd(&chords[  linChannelId * VGRIDSIZE
        //                  + getLinVoxelId(id[0], id[1], id[2], gridN)],
        //          (int)(dimCrossed) * (aDimnext[dim]-aCurr)*length);
        
        //      ... increase chord index (writing index)
        chordId       +=  (int)(dimCrossed);
        
        //      ... update current parameter
        aCurr          = (int)(!dimCrossed) * aCurr
                        + (int)(dimCrossed) * aDimnext[dim];
        //      ... update this axis' paramter to next plane
        aDimnext[dim] +=  (int)(dimCrossed) * aDimup[dim];
        //      ... update this axis' voxel index
        id[dim]       +=  (int)(dimCrossed) * idDimup[dim];

      } // for(dim)
    } // while(aCurr)
  } // for(iRay)
}

#endif  // #ifndef CHORDSCALC_KERNEL