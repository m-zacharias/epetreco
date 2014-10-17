/* TODO Behavior for rays that lie exactly in a plane? Exactly on the
 * intersection line of to orthogonal planes?
 */
#ifndef CHORDSCALC_LOWLEVEL_HPP
#define CHORDSCALC_LOWLEVEL_HPP

#include <cuda.h>
#include <cmath>
#include <algorithm>

#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
#include <iostream>
#endif

/**
 * @brief Struct template: Conditional minimum functor
 * 
 * @tparam n Dimension of argument vectors
 */
template<int n>
struct MinFunctor
{
  /**
   * @brief Member function template: Functor operation
   *
   * @tparam T Type of minimum and possible candidates
   * @arg min_ Result memory for minimum
   * @arg good Result memory for state of result
   * @arg a    Vector of candidates
   * @arg b    Vector of possible candidates
   */
  template<typename T>
  __host__ __device__
  void operator()(
        T * const min_, bool * const good,
        T const a[], bool const b[] )
  {
    // minimum of first n-1 values
    MinFunctor<n-1>()(min_, good, a, b);
    
    // determine min without 'if': sum of products with int-cast exclusive
    // conditions
    *min_ =
          (int)( (*good) &&  b[n-1]) * min((*min_), a[n-1])
         +(int)( (*good) && !b[n-1]) * (*min_)
         +(int)(!(*good) &&  b[n-1]) * a[n-1];
  
    // was any of all n values good?
    *good |= b[n-1];
  }
};


/* Template specialisation for n=2 */
template<>
struct MinFunctor<2>
{
  template<typename T>
  __host__ __device__
  void operator()(
        T * const min_, bool * const good,
        T const a[], bool const b[] )
  {
    // determine min without 'if': sum of products with int-cast exclusive
    // conditions
    *min_ =
          (int)( b[0] &&  b[1]) * min(a[0], a[1])
         +(int)( b[0] && !b[1]) * a[0]
         +(int)(!b[0] &&  b[1]) * a[1];
  
    // was any of the 2 values good?
    *good = b[0] || b[1];
  }
};


/**
 * @brief Struct template: Conditional maximum functor
 * 
 * @tparam n Dimension of argument vectors
 */
template<int n>
struct MaxFunctor
{
  /**
   * @brief Member function template: Functor operation
   *
   * @tparam T Type of maximum and possible candidates
   * @arg max_ Result memory for maximum
   * @arg good Result memory for state of result
   * @arg a    Vector of candidates
   * @arg b    Vector of possible candidates
   */
  template<typename T>
  __host__ __device__
  void operator()(
        T * const max_, bool * const good,
        T const a[], bool const b[] )
  {
    // maximum of first n-1 values
    MaxFunctor<n-1>()(max_, good, a, b);
    
    // determine max without 'if': sum of products with int-cast exclusive
    // conditions
    *max_ =
          (int)( (*good) &&  b[n-1]) * max((*max_), a[n-1])
         +(int)( (*good) && !b[n-1]) * (*max_)
         +(int)(!(*good) &&  b[n-1]) * a[n-1];
  
    // was any of all n values good?
    *good |= b[n-1];
  }
};


/* Template specialisation for n=2 */
template<>
struct MaxFunctor<2>
{
  template<typename T>
  __host__ __device__
  void operator()(
        T * const max_, bool * const good,
        T const a[], bool const b[] )
  {
    // determine max without 'if': sum of products with int-cast exclusive
    // conditions
    *max_ =
          (int)( b[0] &&  b[1]) * max(a[0], a[1])
         +(int)( b[0] && !b[1]) * a[0]
         +(int)(!b[0] &&  b[1]) * a[1];
  
    // was any of the 2 values good?
    *good = b[0] || b[1];
  }
};







template<typename val_t>
__host__ __device__
void
getChords(
//      val_t chords[], int voxelIds[],
//      int const nChords,
//      val_t const ray[],
//      val_t const gridO[], val_t const gridD[], int const gridN[] )
      val_t * const chords, int * const voxelIds,
      int const nChords,
      val_t const * const ray,
      val_t const * const gridO, val_t const * const gridD, int const * const gridN )
{
  // Get intersection minima for all axes, get intersection info
  val_t aDimmin[3];
  val_t aDimmax[3];
  bool  crosses[3];
  getAlphaDimmin(   aDimmin, ray, gridO, gridD, gridN);
  getAlphaDimmax(   aDimmax, ray, gridO, gridD, gridN);
  getCrossesPlanes( crosses, ray, gridO, gridD, gridN);
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
  std::cout << "aDimmin: " << aDimmin[0] << "(" << crosses[0] << "), "
                           << aDimmin[1] << "(" << crosses[1] << "), "
                           << aDimmin[2] << "(" << crosses[2] << ")"
            << std::endl
            << "aDimmax: " << aDimmax[0] << "(" << crosses[0] << "), "
                           << aDimmax[1] << "(" << crosses[1] << "), "
                           << aDimmax[2] << "(" << crosses[2] << ")"
            << std::endl;
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
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
  std::cout << "aMin: " << aMin << "(" << aMinGood << ")"
            << std::endl
            << "aMax: " << aMax << "(" << aMaxGood << ")"
            << std::endl;
#endif
  // Is grid intersected at all, does ray start and end outside the grid?
  // - otherwise return
  if(aMin>aMax || !aMinGood || !aMaxGood) return;
  
  // Get parameter update values 
  val_t aDimup[3];
  getAlphaDimup(  aDimup, ray, gridD);
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
  std::cout << "aDimup: " << aDimup[0] << "(" << crosses[0] << "), "
                          << aDimup[1] << "(" << crosses[1] << "), "
                          << aDimup[2] << "(" << crosses[2] << ")"
            << std::endl;
#endif
  
  // Get id update values
  int idDimup[3];
  getIdDimup( idDimup, ray);
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
  std::cout << "idDimup: " << idDimup[0] << ", "
                           << idDimup[1] << ", "
                           << idDimup[2]
            << std::endl;
#endif
  
  // Initialize array of next parameters
  val_t aDimnext[3];
  for(int dim=0; dim<3; dim++) aDimnext[dim] = aDimmin[dim] + aDimup[dim];
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
  std::cout << "aDimnext: " << aDimnext[0] << ", "
                            << aDimnext[1] << ", "
                            << aDimnext[2]
            << std::endl;
#endif
  
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
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
  std::cout << "id: " << id[0] << ", "
                      << id[1] << ", "
                      << id[2]
            << std::endl;
#endif

  // Initialize current parameter
  val_t aCurr = aMin;
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
  std::cout << "aCurr: " << aCurr
            << std::endl;
#endif

  // Get length of ray
  val_t const length(getLength(ray));
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
  std::cout << "length: " << length
            << std::endl;
#endif
  
 
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
  std::cout << std::endl
            << "### ITERATIONS"       << std::endl
            << "####################" << std::endl;
#endif
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
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
      std::cout << "chordId: " << chordId
                << std::endl
                << "aCurr: " << aCurr
                << std::endl
                << "aNext: " << aNext
                << std::endl
                << "aDimnext[" << dim << "]: " << aDimnext[dim]
                << std::endl;
#endif
      // Is this axis' plane crossed at the next parameter of intersection?
      bool dimCrossed = (aDimnext[dim] == aNext);
      anyAxisCrossed |= dimCrossed;
      

      // If this axis' plane is crossed ...
      //      ... clear and write chord length and voxel index
      chords[     chordId]
              *= (int)(!dimCrossed);
      chords[     chordId]
              += (int)( dimCrossed) * (aDimnext[dim]-aCurr)*length;
      for(int writeDim=0; writeDim<3; writeDim++)
      {
        voxelIds[ chordId*3 + writeDim]
              *= (int)(!dimCrossed);
        voxelIds[ chordId*3 + writeDim]
              += (int)( dimCrossed) * id[writeDim];
      }
      
      //      ... increase chord index (write index)
      chordId       +=  (int)(dimCrossed);
      
      //      ... update current parameter
      aCurr          = (int)(!dimCrossed) * aCurr
                      + (int)(dimCrossed) * aDimnext[dim];
      //      ... update this axis' paramter to next plane
      aDimnext[dim] +=  (int)(dimCrossed) * aDimup[dim];
      //      ... update this axis' voxel index
      id[dim]       +=  (int)(dimCrossed) * idDimup[dim];
      
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
      std::cout << "chord: " << chords[chordId-(int)(dimCrossed)]
                << std::endl
                << "--- dim = " << dim << " end " << "----"
                << std::endl
                << "--------------------"
                << std::endl;
#endif
    }
#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
    std::cout << "////////////////////"
              << std::endl << std::endl;
#endif
    
//    if(!anyAxisCrossed) throw -1;
  }
}


/**
 * @brief Function template that for a given ray and a given grid determines for
 *        all axis, if the planes orthogonal to that axis are intersected by the
 *        ray.
 *
 * @arg crosses   Result memory
 * @arg ray       Array of ray start and end coordinates
 * @arg gridO     Array of grid origin coordinates
 * @arg gridD     Array of grid plane spacings for each axis
 * @arg gridN     Array of grid voxel numbers for each axis
 */
template<typename val_t>
__host__ __device__
void
getCrossesPlanes(
      bool crosses[],
      val_t const ray[],
      val_t const gridO[], val_t const gridD[], int const gridN[] )
{
  bool goodEvenIfParallel = true;
  
  for(int dim=0; dim<3; dim++)
  {
    crosses[dim] = (ray[dim+3] - ray[dim] != 0);
    // Is ray orthogonal to axis? If so, is ray's coordinate on this axis inside
    // grid?
    goodEvenIfParallel &= (
          crosses[dim]
     || (!crosses[dim] && ray[dim]>=gridO[dim]
                       && ray[dim]< gridO[dim]+gridD[dim]*gridN[dim]
        )
                          );
  }
  // If for any orthogonal axis the ray's coordinate does not lie in grid: No
  // planes crossed at all!
  for(int dim=0; dim<3; dim++)
    crosses[dim] &= goodEvenIfParallel;
}


/**
 * @brief Function template that for a given ray and a given grid determines
 *        for one grid plane the ray parameter of the intersection point of the
 *        ray with that plane.
 *        The ray parameter gives the linear position on
 *        the ray, 0 corresponds to the start point and 1 corresponds to the end
 *        point.
 *        The function call is also safe in cases, where ray and plane do not
 *        intersect (no division by zero) but then the return value is
 *        meaningless, of course.
 *
 * @arg i         Index of plane in dim
 * @arg dim       Axis perpendicular to plane
 * @arg ray       Array of ray start and end coordinates
 * @arg gridO     Array of grid origin coordinates
 * @arg gridD     Array of grid plane spacings for each axis
 * @arg gridN     Array of grid voxel numbers for each axis
 */
template<typename val_t>
__host__ __device__
val_t
alphaFromId(
      int const i, int const dim,
      val_t const ray[],
      val_t const gridO[], val_t const gridD[], int const gridN[] )
{
  val_t divisor = ray[dim+3] - ray[dim];
  divisor += (divisor==0);

  return ( gridO[dim] + i * gridD[dim] - ray[dim] )
        /  divisor;
} 


/**
 * @brief Function template that for a given ray and a given grid and axis
 *        determines from the ray parameter the corresponding (continuous)
 *        value in this axis' plane index space.
 *        The ray parameter gives the linear position on
 *        the ray, 0 corresponds to the start point and 1 corresponds to the end
 *        point.
 *
 * @arg alpha     Ray parameter
 * @arg dim       Axis perpendicular to plane
 * @arg ray       Array of ray start and end coordinates
 * @arg gridO     Array of grid origin coordinates
 * @arg gridD     Array of grid plane spacings for each axis
 * @arg gridN     Array of grid voxel numbers for each axis
 */
template<typename val_t>
__host__ __device__
val_t
phiFromAlpha(
      val_t const alpha,
      int const dim,
      val_t const ray[],
      val_t const gridO[], val_t const gridD[], int const gridN[] )
{
  return ( ray[dim] + alpha * (ray[dim+3] - ray[dim]) - gridO[dim] )
         /
         ( gridD[dim] );
} 


/**
 * @brief Function template that for a given ray and a given grid determines for
 *        each axis the minimum of all parameters of intersections of the ray
 *        with grid planes that are orthogonal to this axis.
 *        If an axis' orthogonal planes are not intersected, the result is
 *        meaningles for that axis only.
 *        'alphaFromId' must not return 'nan' but an arbitrary valid number for
 *        axis whose orthogonal planes are not intersected!!!
 *
 * @tparam val_t  Coordinate type
 * @arg aDimmin   Result memory for mimimum intersection parameters
 * @arg ray       Array of ray start and end coordinates
 * @arg gridO     Array of grid origin coordinates
 * @arg gridD     Array of grid plane spacings for each axis
 * @arg gridN     Array of grid voxel numbers for each axis
 */
template<typename val_t>
__host__ __device__
void
getAlphaDimmin( 
      val_t aDimmin[],
      val_t const ray[],
      val_t const gridO[], val_t const gridD[], int const gridN[] )
{
  for(int dim=0; dim<3; dim++)
  {
    aDimmin[dim] = min(
          alphaFromId(0,          dim, ray, gridO, gridD, gridN),
          alphaFromId(gridN[dim], dim, ray, gridO, gridD, gridN)
                           );
  }
} 


/**
 * @brief Function template that for a given ray and a given grid determines for
 *        each axis the maximum of all parameters of intersections of the ray
 *        with grid planes that are orthogonal to this axis.
 *        If an axis' orthogonal planes are not intersected, the result is
 *        meaningles for that axis only.
 *        'alphaFromId' must not return 'nan' but an arbitrary valid number for
 *        axis whose orthogonal planes are not intersected!!!
 *
 * @tparam val_t  Coordinate type
 * @arg aDimmax   Result memory for mimimum intersection parameters
 * @arg ray       Array of ray start and end coordinates
 * @arg gridO     Array of grid origin coordinates
 * @arg gridD     Array of grid plane spacings for each axis
 * @arg gridN     Array of grid voxel numbers for each axis
 */
template<typename val_t>
__host__ __device__
void
getAlphaDimmax(
      val_t aDimmax[],
      val_t const ray[],
      val_t const gridO[], val_t const gridD[], int const gridN[] )
{
  for(int dim=0; dim<3; dim++)
  {
     aDimmax[dim] = max(
          alphaFromId(0,          dim, ray, gridO, gridD, gridN),
          alphaFromId(gridN[dim], dim, ray, gridO, gridD, gridN)
                            );
  }
} 


/**
 * @brief Function template that from the array of ray parameter minima for
 *        each axis determines the one that corresponds to the point where the
 *        ray enters the grid.
 *
 * @arg aMin          Result memory for minimum parameter of ray in grid
 * @arg good          Memory for the state of the result
 * @arg aDimmin       Array of intersection minima for each axis
 * @arg aDimminGood   Array of states of the intersection minima for each axis
 */
template<typename val_t>
__host__ __device__
void
getAlphaMin(
      val_t * aMin, bool * good,
      val_t const aDimmin[], bool const aDimminGood[] )
{
  // Max of mins!!!
  MaxFunctor<3>()(aMin, good, aDimmin, aDimminGood);
} 


/**
 * @brief Function template that from the array of ray parameter maxima for
 *        each axis determines the one that corresponds to the point where the
 *        ray leaves the grid.
 *
 * @arg aMax          Result memory for maximum parameter of ray in grid
 * @arg good          Memory for the state of the result
 * @arg aDimmax       Array of intersection maxima for each axis
 * @arg aDimmaxGood   Array of states of the intersection maxima for each axis
 */
template<typename val_t>
__host__ __device__
void
getAlphaMax(
      val_t * aMax, bool * good,
      val_t const aDimmax[], bool const aDimmaxGood[] )
{
  // Min of maxs!!!
  MinFunctor<3>()(aMax, good, aDimmax, aDimmaxGood);
}


/**
 * @brief Function template that returns the length of a given ray.
 *
 * @arg ray   Array of ray start and end coordinates
 */
template<typename val_t>
__host__ __device__
val_t
getLength(
      val_t const ray[])
{
  return sqrt(
        (ray[0+3] - ray[0]) * (ray[0+3] - ray[0])
       +(ray[1+3] - ray[1]) * (ray[1+3] - ray[1])
       +(ray[2+3] - ray[2]) * (ray[2+3] - ray[2])
                  );
}


/**
 * @brief Function template that for a given ray and a given grid determines the
 *        the ray parameter increase that corresponds to advancing from plane n
 *        to plane n+1 for all axes.
 *
 * @arg aDimup  Result memory
 * @arg ray     Array of ray start and end coordinates
 * @arg gridD   Array of grid plane spacings for each axis
 */
template<typename val_t>
__host__ __device__
void
getAlphaDimup(
      val_t aDimup[],
      val_t const ray[],
      val_t const gridD[] )
{
  
  for(int dim=0; dim<3; dim++)
  {
    // Make sure, no division by zero will be performed!
    val_t divisor = sqrt((ray[dim+3]-ray[dim])*(ray[dim+3]-ray[dim]));
    divisor += (int)(divisor==0);
    
    aDimup[dim] =  gridD[dim] / divisor;
  }
}


/**
 *@brief Function template that for a given ray determines the voxel index
 *       update value for all axes. If increasing alpha corresponds to
 *       increasing plane indices for an axis, this axis' update value is +1.
 *       Otherwise, ist is -1.
 *
 * @arg idDimup  Result memory
 * @arg ray      Array of ray start and end coordinates
 */
template<typename val_t>
__host__ __device__
void
getIdDimup(
      int idDimup[],
      val_t const ray[] )
{
  for(int dim=0; dim<3; dim++)
  {
    idDimup[dim] = (int)(ray[dim] <  ray[dim+3])
                  -(int)(ray[dim] >= ray[dim+3]);
  }
}

#endif  // #ifndef CHORDSCALC_LOWLEVEL_HPP
