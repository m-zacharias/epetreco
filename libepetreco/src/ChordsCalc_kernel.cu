#include "ChordsCalc_lowlevel.hpp"

#define THREAD_BUFFER_SIZE 8
#define LINID( idx, idy, idz ) {}

__global__
void chordsCalc(
      val_t * const chords, int * voxelIds,
      val_t const * const ray,
      val_t const * gridO, val_t const * const gridD, int const * const gridN )
{
  // ##################
  // ### INITIALIZATION
  // ##################

  // Get intersection minima for all axes, get intersection info
  val_t aDimmin[3];
  val_t aDimmax[3];
  bool  crosses[3];
  getAlphaDimmin(   aDimmin, ray, gridO, gridD, gridN);
  getAlphaDimmax(   aDimmax, ray, gridO, gridD, gridN);
  getCrossesPlanes( crosses, ray, gridO, gridD, gridN);
  
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
  if(aMin>aMax || !aMinGood || !aMaxGood) return;

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
  
  // Initialize local double buffers
  val_t * localChords0[THREAD_BUFFER_SIZE];
  val_t * localChords1[THREAD_BUFFER_SIZE];
  val_t * localChords = localChords0;

  int * localChordIds0[THREAD_BUFFER_SIZE*3]
  int * localChordIds1[THREAD_BUFFER_SIZE*3]
  int * localChordIds = localChordIds0;
  
  
  
  // ##################
  // ###  ITERATIONS
  // ##################
  int chordId = 0;
  while(aCurr < aMax)
  {
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
      localChords[     chordId%THREAD_BUFFER_SIZE]
              *= (int)(!dimCrossed);
      localChords[     chordId%THREAD_BUFFER_SIZE]
              += (int)( dimCrossed) * (aDimnext[dim]-aCurr)*length;
      for(int writeDim=0; writeDim<3; writeDim++)
      {
        localVoxelIds[(chordId%THREAD_BUFFER_SIZE)*3 + writeDim]
              *= (int)(!dimCrossed);
        localVoxelIds[(chordId%THREAD_BUFFER_SIZE)*3 + writeDim]
              += (int)( dimCrossed) * id[writeDim];
      }
      
      //      ... increase chord index (writing index)
      chordId       +=  (int)(dimCrossed);
      
      //      ... update current parameter
      aCurr          = (int)(!dimCrossed) * aCurr
                      + (int)(dimCrossed) * aDimnext[dim];
      //      ... update this axis' paramter to next plane
      aDimnext[dim] +=  (int)(dimCrossed) * aDimup[dim];
      //      ... update this axis' voxel index
      id[dim]       +=  (int)(dimCrossed) * idDimup[dim];
      
      // 
      if(chordId%THREAD_BUFFER_SIZE == THREAD_BUFFER_SIZE-1 || aNext == aMax)
      {
        int write_size = chordId%THREAD_BUFFER_SIZE + 1;
        int start = chordId - write_size + 1;
        int end   = chordId + 1;

        if(localChords == localChords0)
        {
          localChords   = localChords1;
          localChordIds = localChordIds1;
          
//          memcpy((void*)&chords[     chordId-THREAD_BUFFER_SIZE+1 ],
//                 (void*)localChords0,     THREAD_BUFFER_SIZE*sizeof(val_t));
//          memcpy((void*)&chordIds[3*(chordId-THREAD_BUFFER_SIZE+1)],
//                 (void*)localChordIds0, 3*THREAD_BUFFER_SIZE*sizeof(val_t));
          for(int i=start; i<end; i++)
          {
            int idx = localChordIds0[3*(i%THREAD_BUFFER_SIZE)+0];
                      localChordIds0[3*(i%THREAD_BUFFER_SIZE)+0] = 0;
            int idy = localChordIds0[3*(i%THREAD_BUFFER_SIZE)+1];
                      localChordIds0[3*(i%THREAD_BUFFER_SIZE)+1] = 0;
            int idz = localChordIds0[3*(i%THREAD_BUFFER_SIZE)+2];
                      localChordIds0[3*(i%THREAD_BUFFER_SIZE)+2] = 0;

            int voxelId = LINID(idx, idy, idz);
            
            chords[voxelId] = localChords0[i%THREAD_BUFFER_SIZE];
                              localChords0[i%THREAD_BUFFER_SIZE] = 0.;
          }
        }
        else if( localChords == localChords1)
        {
          localChords   = localChords0;
          localChordIds = localChordIds0;
          
//          memcpy((void*)&chords[     chordId-THREAD_BUFFER_SIZE+1 ],
//                 (void*)localChords1,     THREAD_BUFFER_SIZE*sizeof(val_t));
//          memcpy((void*)&chordIds[3*(chordId-THREAD_BUFFER_SIZE+1)],
//                 (void*)localChordIds1, 3*THREAD_BUFFER_SIZE*sizeof(val_t));
          for(int i=start; i<end; i++)
          {
            int idx = localChordIds1[3*(i%THREAD_BUFFER_SIZE)+0];
                      localChordIds1[3*(i%THREAD_BUFFER_SIZE)+0] = 0;
            int idy = localChordIds1[3*(i%THREAD_BUFFER_SIZE)+1];
                      localChordIds1[3*(i%THREAD_BUFFER_SIZE)+1] = 0;
            int idz = localChordIds1[3*(i%THREAD_BUFFER_SIZE)+2];
                      localChordIds1[3*(i%THREAD_BUFFER_SIZE)+2] = 0;

            int voxelId = LINID(idx, idy, idz);
            
            chords[voxelId] = localChords1[i%THREAD_BUFFER_SIZE];
                              localChords1[i%THREAD_BUFFER_SIZE] = 0.;
          }
        }
      }
    }
  }
}
