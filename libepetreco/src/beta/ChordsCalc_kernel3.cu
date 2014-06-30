#ifndef CHORDSCALC_KERNEL
#define CHORDSCALC_KERNEL

#define DEBUG_MACRO ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))

#include "ChordsCalc_lowlevel.hpp"
#include "CUDA_HandleError.hpp"
//#include "MeasurementSetup.hpp"

#include <math.h>
#include <curand_kernel.h>
#include <cstdio>
#include <assert.h>

#ifndef PRINT_KERNEL
#define PRINT_KERNEL 0
#endif


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
//{
//  return  idz * gridN[0]*gridN[1]
//        + idy * gridN[0]
//        + idx;
//}
{
  return  idz
        + idy *          gridN[2]
        + idx * gridN[1]*gridN[2];
}



__inline__ __host__ __device__
int getLinMtxId(
      int const rowId, int const rowDim,
      int const colId, int const colDim )
{
  return colId * colDim + rowId;
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
template<typename T>
__host__ __device__
void getTransformation(
      T * const trafo,
      T const * const pos, T const * const edges,
      T const sin, T const cos)
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
template<typename T, typename ConcreteMeasurementSetup>
__device__
void getRay(
      T * const                         ray,
      curandState &                     state,
      int const * const                 dimChannelId,
      ConcreteMeasurementSetup const &  setup )
{
#if DEBUG_MACRO
/**/int id = blockDim.x * blockIdx.x + threadIdx.x;
#endif

  // Get geometrical properties of the channel
  T pos0[3];
  T pos1[3];
  T edges[3];
  T sin, cos;
  setup.getGeomProps(pos0, pos1, edges, &sin, &cos, dimChannelId);
#if DEBUG_MACRO
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
  T trafo0[12];
  getTransformation(trafo0, pos0, edges, sin, cos);
#if DEBUG_MACRO
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

  T trafo1[12];
  getTransformation(trafo1, pos1, edges, sin, cos);
#if DEBUG_MACRO
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
  T rand[8];
  for(int i=0; i<3; i++)
  {
    rand[i]   = curand_uniform(&state);
    rand[i+4] = curand_uniform(&state);
  }
  rand[3] = 1.;
  rand[7] = 1.;
#if DEBUG_MACRO
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
  for(int rowId=0; rowId<3; rowId++)
  {
    ray[rowId]   = 0.;
    ray[rowId+3] = 0.;

    for(int colId=0; colId<4; colId++)
    {
      ray[rowId]   += trafo0[rowId*4 + colId] * rand[colId];
      ray[rowId+3] += trafo1[rowId*4 + colId] * rand[colId+4];
    }
  }
#if DEBUG_MACRO
/**/if(id == PRINT_KERNEL)
/**/{
/**/  printf("    -----\n");
/**/  printf("    start: %05.3f,  %05.3f,  %05.3f\n",
/**/          ray[0], ray[1], ray[2]); 
/**/  printf("      end: %05.3f,  %05.3f,  %05.3f\n",
/**/          ray[3], ray[4], ray[5]); 
/**/}
#endif
}




/**
 * @brief Initialize chordsCalc kernel function.
 *
 * - set globalId
 * - set rowId
 * - set linChannelId
 * - set dimChannelId
 * - initialize kernelRandState
 */
template<
      typename ConcreteMeasurementSetup
    , typename Event
>
__device__ __forceinline__
bool _chordsCalcInitKernel
(
      int &                             globalId,
      int &                             rowId,
      int &                             linChannelId,
      curandState &                     kernelRandState,
      int * const                       dimChannelId, 

      int const                         nChannels,

      ConcreteMeasurementSetup const *  setup,
      Event const * const               y,
      int const                         randomSeed
)
{
  /* Global id of channel */
  globalId     = blockDim.x * blockIdx.x + threadIdx.x;
  rowId        = blockIdx.x;          // index of row in current system
                                      //   matrix chunk (!)
  linChannelId = y[rowId].channel();  // global (!) linearized channel index 
                                      //   - read explictly from current
                                      //   measurement chunk
  
  /* If this kernel got no valid data to work with: return false */
  if(linChannelId >= nChannels || linChannelId < 0)
    return false;

#if DEBUG_MACRO
/**/if(globalId == PRINT_KERNEL)
/**/{
/**/  printf("\nchordsCalc(...):\n");
/**/}
#endif

  /* Initialize random state (for making rays) */
  curand_init(randomSeed, linChannelId, 0, &kernelRandState);
  
  /* Get geometrical indices of channel */
  setup->sepChannelId(dimChannelId, linChannelId);

#if DEBUG_MACRO
/**/if(globalId == PRINT_KERNEL)
/**/{
/**/  printf("\n");
/**/  printf("linChannelId: %i,  dimChannelId: %i,  %i,  %i,  %i,  %i\n",
/**/         linChannelId, dimChannelId[0], dimChannelId[1], dimChannelId[2],
/**/         dimChannelId[3], dimChannelId[4]);
/**/}
#endif
  
  /* Return: kernel successfully initialized, got valid data to work with */
  return true;
}



/**
 * @brief Initialize variables for Siddon algorithm.
 * 
 * - determine which planes are crossed (crosses)
 * - calculate length of ray (length)
 * - calculate alpha parameter update values for each dimension (aDimup)
 * - calculate plane index update values for each dimension (idDimup)
 * - initialize next alpha values for each dimension (aDimnext)
 * - initialize array of current voxel's indices (id)
 * - determine alpha parameter of next intersection (aNext, aNextExists)
 * - initialize current alpha parameter (aCurr)
 */
template<
      typename T
>
__device__ __forceinline__
void _chordsCalcInitSiddon
(
      bool      crosses[3],
      T &       length,
      T         aDimup[3],
      int       idDimup[3],
      T         aDimnext[3],
      int       id[3],
      T &       aNext,
      bool &    aNextExists,
      T &       aCurr,
      
      T const   ray[6],
      T const   gridO[3],
      T const   gridD[3],
      int const gridN[3]
)
{
#if DEBUG_MACRO
/**/if(globalId == PRINT_KERNEL)
/**/{
/**/  printf("\n");
/**/  printf("### INITIALIZATION\n");
/**/}
#endif

  // Get intersection minima for all axes, get intersection info
//#if DEBUG_MACRO
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
  T aDimmin[3];
  T aDimmax[3];
  getAlphaDimmin(   aDimmin, ray, gridO, gridD, gridN);
  getAlphaDimmax(   aDimmax, ray, gridO, gridD, gridN);
  getCrossesPlanes( crosses, ray, gridO, gridD, gridN);
//#if DEBUG_MACRO
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
  T     aMin;
  T     aMax;
  bool  aMinGood;
  bool  aMaxGood;
  getAlphaMin(  &aMin, &aMinGood, aDimmin, crosses);
  getAlphaMax(  &aMax, &aMaxGood, aDimmax, crosses);
  
  // Do entry and exit points lie in beween ray start and end points?
  aMinGood &= (aMin >= 0. && aMin <= 1.);
  aMaxGood &= (aMax >= 0. && aMax <= 1.);
  
  // Is grid intersected at all, does ray start and end outside the grid?
  // - otherwise return
  if(aMin>aMax || !aMinGood || !aMaxGood) return;

  // Get length of ray
  length = getLength(ray);
  
  // Get parameter update values 
  getAlphaDimup(  aDimup, ray, gridD);
  
  // Get id update values
  getIdDimup( idDimup, ray);
  
  // Initialize array of next parameters
  for(int dim=0; dim<3; dim++)
  {
    aDimnext[dim] = aDimmin[dim];
    while(aDimnext[dim]<=aMin)
      aDimnext[dim] += aDimup[dim];
  }

  // Initialize array of voxel indices
  MinFunctor<3>()(&aNext, &aNextExists, aDimnext, crosses);

#if DEBUG_MACRO
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
          T(0.5)*(aMin + aNext), dim, ray, gridO, gridD, gridN
                                     )
                        );
#if DEBUG_MACRO
/**/  if(globalId == PRINT_KERNEL)
/**/  {
/**/    printf("phiFromAlpha: %f  ",
/**/           phiFromAlpha(float(0.5)*(aMin + aNext), dim, ray, gridO, gridD, gridN));
/**/    printf("id[%i]: %02i  ",
/**/           dim, id[dim]);
/**/  }
#endif
  } // for(dim)
#if DEBUG_MACRO
/**/  if(globalId == PRINT_KERNEL)
/**/  {
/**/    printf("\n");
/**/  }
#endif

  // Initialize current parameter
  aCurr = aMin;

#if DEBUG_MACRO
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
}



template<
      typename T
>
__device__ __forceinline__
void _chordsCalcSiddonCycles
(
      T *         chords,
      T &         aCurr,
      T &         aNext,
      bool &      aNextExists,
      T           aDimnext[3],
      int         id[3],

      int const   vgridSize,
      int const   chunkSize,
      int const   rowId,
      bool const  crosses[3],
      T const     aDimup[3],
      int const   idDimup[3],
      int const   gridN[3],
      T const     length
)
{
//  /* Print arguments and return */
//  if(blockIdx.x*blockDim.x+threadIdx.x == 0)
//  {
//    printf("aCurr: %5.3f, aNext: %5.3f, aNextExists: %i\n",
//           aCurr, aNext, aNextExists);
//    printf("aDimnext: %5.3f, %5.3f, %5.3f\n",
//           aDimnext[0], aDimnext[1], aDimnext[2]);
//    printf("id: %i, %i, %i\n",
//           id[0], id[1], id[2]);
//    printf("vgridSize: %i, chunkSize: %i, rowId: %i\n",
//           vgridSize, chunkSize, rowId);
//    printf("crosses: %i, %i, %i\n",
//           crosses[0], crosses[1], crosses[2]);
//    printf("aDimup: %5.3f, %5.3f, %5.3f\n",
//           aDimup[0], aDimup[1], aDimup[2]);
//    printf("idDimup: %i, %i, %i\n",
//           idDimup[0], idDimup[1], idDimup[2]);
//    printf("gridN: %i, %i, %i\n",
//           gridN[0], gridN[1], gridN[2]);
//    printf("length: %5.3f\n\n",
//           length);
//    return;
//  }

#if DEBUG_MACRO
/**/if(globalId == PRINT_KERNEL)
/**/{
/**/  printf("\n");
/**/  printf("### ITERATIONS\n");
/**/}
#endif
  while(   id[0]<gridN[0]
        && id[1]<gridN[1]
        && id[2]<gridN[2]
        && id[0]>=0
        && id[1]>=0
        && id[2]>=0)
  {
#if DEBUG_MACRO
/**/if(globalId == PRINT_KERNEL)
/**/{
///**/  printf("aCurr: %05.2f\n", aCurr);
/**/  printf("aCurr: %f\n", aCurr);
/**/  printf("aCurr-aMax: %e\n", aCurr-aMax);
/**/  printf("aCurr<aMax: %i\n", aCurr<aMax);
/**/}
#endif

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
#if DEBUG_MACRO
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
      
      int rowDim = vgridSize;
      int colId  = getLinVoxelId(id[0], id[1], id[2], gridN);
      int colDim = chunkSize;
      int linMtxId = getLinMtxId(rowId, rowDim, colId, colDim);
      
///* !!! */syncthreads();
///* !!! */if(linMtxId>=rowDim*colDim)
/////* !!! */if(blockIdx.x*blockDim.x+threadIdx.x == 0)
///* !!! */{
///* !!! */  printf("kernel: %i, block: %i",
///* !!! */         threadIdx.x, blockIdx.x);
///* !!! */  printf("id[0]: %i, id[1]: %i, id[2]: %i\n",
///* !!! */         id[0], id[1], id[2]);
///* !!! */  printf("rowDim: %i, colId: %i, colDim: %i, rowId: %i\n",
///* !!! */         rowDim, colId, colDim, rowId);
///* !!! */  printf("linMtxId: %i\n\n", linMtxId);
///* !!! */}
///* !!! */syncthreads();
////      if((threadIdx.x == 0) && (blockIdx.x == 114))
////        printf("linMtxId: %i\n", linMtxId);

//      assert(linMtxId<=chunkSize*vgridSize);

      atomicAdd(&chords[linMtxId],
                (int)(dimCrossed) * (aDimnext[dim]-aCurr)*length);

//      atomicAdd(&chords[linMtxId], 1.);

      //atomicAdd(&chords[  linChannelId * VGRIDSIZE
      //                  + getLinVoxelId(id[0], id[1], id[2], gridN)],
      //          (int)(dimCrossed) * (aDimnext[dim]-aCurr)*length);
      
      //      ... update current parameter
      aCurr          = (int)(!dimCrossed) * aCurr
                      + (int)(dimCrossed) * aDimnext[dim];
      //      ... update this axis' paramter to next plane
      aDimnext[dim] +=  (int)(dimCrossed) * aDimup[dim];
      //      ... update this axis' voxel index
      id[dim]       +=  (int)(dimCrossed) * idDimup[dim];

      if(id[dim] >= gridN[dim]) break;
      if(id[dim] <  0)          break;
    } // for(dim)
  } // while(aCurr)
}      



template<
      typename T
    , typename Event
    , typename ConcreteMeasurementSetup
>
__global__
void chordsCalc(
      T * const                         chords,
      T * const                         rays,
      Event const * const               y,
      T const * const                   gridO,
      T const * const                   gridD,
      int const * const                 gridN,
      int const                         channelOffset,
      int const                         nChannels,
      int const                         chunkSize,
      int const                         vgridSize,
      ConcreteMeasurementSetup const *  setup,
      int const                         randomSeed,
      int const                         nThreadRays )
{
  // ##############################
  // ### INITIALIZE KERNEL INSTANCE
  // ##############################

  int           globalId; 
  int           rowId;
  int           linChannelId;
  curandState   kernelRandState;
  int           dimChannelId[5]; 
  
  if(!_chordsCalcInitKernel(
            globalId, 
            rowId,
            linChannelId,
            kernelRandState,
            dimChannelId,

            nChannels,

            setup,
            y,
            randomSeed
      ))
    return;

  T ray[6];
  for(int iRay=0; iRay<nThreadRays; iRay++)
  {
    // Get ray
    getRay(ray, kernelRandState, dimChannelId, *setup);

    // Write ray 
    for(int dim=0; dim<6; dim++)
      //rays[6*(linChannelId*nThreadRays + iRay) + dim] = ray[dim];
      rays[6*(rowId*nThreadRays + iRay) + dim] = ray[dim];

#if DEBUG_MACRO
/**/if(globalId == PRINT_KERNEL)
/**/{
/**/  printf("\n");
/**/  printf("start:  %06.3f,  %06.3f,  %06.3f\n",
/**/          ray[0], ray[1], ray[2]);
/**/  printf("end  :  %06.3f,  %06.3f,  %06.3f\n",
/**/          ray[3], ray[4], ray[5]);
/**/}
#endif


    // #########################################
    // ### INITIALIZE SIDDON ALGORITHM VARIABLES
    // #########################################
    
    bool  crosses[3];
    T     length;
    T     aDimup[3];
    int   idDimup[3];
    T     aDimnext[3];
    int   id[3];
    T     aNext;
    bool  aNextExists;
    T     aCurr;
    
    _chordsCalcInitSiddon(
          crosses,
          length,
          aDimup,
          idDimup,
          aDimnext,
          id,
          aNext,
          aNextExists,
          aCurr,
          
          ray,
          gridO,
          gridD,
          gridN
    );
  

    // #####################################################
    // ### SIDDON ALGORITHM CYCLES (FOLLOW RAY THROUGH GRID)
    // #####################################################
   
    _chordsCalcSiddonCycles(
          chords,
          aCurr,
          aNext,
          aNextExists,
          aDimnext,
          id,
          
          vgridSize,
          chunkSize,
          rowId,
          crosses,
          aDimup,
          idDimup,
          gridN,
          length
    );

  } // for(iRay)
}



template<
      typename T
    , typename Event
    , typename ConcreteMeasurementSetup
>
__global__
void chordsCalc_noVis(
      T * const                         chords,
      Event const * const               y,
      T const * const                   gridO,
      T const * const                   gridD,
      int const * const                 gridN,
      int const                         channelOffset,
      int const                         nChannels,
      int const                         chunkSize,
      int const                         vgridSize,
      ConcreteMeasurementSetup const *  setup,
      int const                         randomSeed,
      int const                         nThreadRays )
{
  // ##############################
  // ### INITIALIZE KERNEL INSTANCE
  // ##############################

  int           globalId; 
  int           rowId;
  int           linChannelId;
  curandState   kernelRandState;
  int           dimChannelId[5]; 
  
  if(!_chordsCalcInitKernel(
            globalId, 
            rowId,
            linChannelId,
            kernelRandState,
            dimChannelId,

            nChannels,

            setup,
            y,
            randomSeed
      ))
    return;

  T ray[6];
  for(int iRay=0; iRay<nThreadRays; iRay++)
  {
    // Get ray
    getRay(ray, kernelRandState, dimChannelId, *setup);

#if DEBUG_MACRO
/**/if(globalId == PRINT_KERNEL)
/**/{
/**/  printf("\n");
/**/  printf("start:  %06.3f,  %06.3f,  %06.3f\n",
/**/          ray[0], ray[1], ray[2]);
/**/  printf("end  :  %06.3f,  %06.3f,  %06.3f\n",
/**/          ray[3], ray[4], ray[5]);
/**/}
#endif


    // #########################################
    // ### INITIALIZE SIDDON ALGORITHM VARIABLES
    // #########################################
    
    bool  crosses[3];
    T     length;
    T     aDimup[3];
    int   idDimup[3];
    T     aDimnext[3];
    int   id[3];
    T     aNext;
    bool  aNextExists;
    T     aCurr;
    
    _chordsCalcInitSiddon(
          crosses,
          length,
          aDimup,
          idDimup,
          aDimnext,
          id,
          aNext,
          aNextExists,
          aCurr,
          
          ray,
          gridO,
          gridD,
          gridN
    );
  

    // #####################################################
    // ### SIDDON ALGORITHM CYCLES (FOLLOW RAY THROUGH GRID)
    // #####################################################
   
    _chordsCalcSiddonCycles(
          chords,
          aCurr,
          aNext,
          aNextExists,
          aDimnext,
          id,
          
          vgridSize,
          chunkSize,
          rowId,
          crosses,
          aDimup,
          idDimup,
          gridN,
          length
    );
  } // for(iRay)
}



template<typename T, typename G, typename S, typename EventVector>
void chordsCalc(
      int const           chunkId,
      int const           nChannels,
      int const           chunkSize,
      int const           nThreads,
      T * const           chords,
      T * const           rays,
      EventVector * const y,
      G * const           grid,
      int const           vgridSize,
      S * const           setup,
      int const           randomSeed,
      int const           nThreadRays )
{
  chordsCalc<<<chunkSize, nThreads>>>(
        chords, rays,
        static_cast<typename EventVector::elem_t *>(y->data()),
        grid->deviRepr()->gridO,
        grid->deviRepr()->gridD,
        grid->deviRepr()->gridN,
        chunkId*chunkSize, nChannels, chunkSize, vgridSize,
        setup->deviRepr(),
        randomSeed,
        nThreadRays);
  HANDLE_ERROR( cudaGetLastError() );
}



template<typename T, typename G, typename S, typename EventVector>
void chordsCalc_noVis(
      int const           chunkId,
      int const           nChannels,
      int const           chunkSize,
      int const           nThreads,
      T * const           chords,
      EventVector * const y,
      G * const           grid,
      int const           vgridSize,
      S * const           setup,
      int const           randomSeed,
      int const           nThreadRays )
{
  chordsCalc_noVis<<<chunkSize, nThreads>>>(
        chords,
        static_cast<typename EventVector::elem_t *>(y->data()),
        grid->deviRepr()->gridO,
        grid->deviRepr()->gridD,
        grid->deviRepr()->gridN,
        chunkId*chunkSize, nChannels,
        chunkSize, vgridSize,
        setup->deviRepr(),
        randomSeed,
        nThreadRays);
  HANDLE_ERROR( cudaGetLastError() );
}

#undef DEBUG_MACRO

#endif  // #ifndef CHORDSCALC_KERNEL
