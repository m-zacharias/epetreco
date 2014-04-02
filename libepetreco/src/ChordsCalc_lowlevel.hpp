#ifndef CHORDSCALC_LOWLEVEL_HPP
#define CHORDSCALC_LOWLEVEL_HPP

//#include "Siddon_helper.hpp"
#include <cmath>
#include "FileTalk.hpp"

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
  void operator()(
        T * const min_, bool * const good,
        T const a[], bool const b[] )
  {
    // minimum of first n-1 values
    MinFunctor<n-1>()(min_, good, a, b);
    
    // determine min without 'if': sum of products with int-cast exclusive
    // conditions
    *min_ =
          (int)( (*good) &&  b[n-1]) * std::min((*min_), a[n-1])
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
  void operator()(
        T * const min_, bool * const good,
        T const a[], bool const b[] )
  {
    // determine min without 'if': sum of products with int-cast exclusive
    // conditions
    *min_ =
          (int)( b[0] &&  b[1]) * std::min(a[0], a[1])
         +(int)( b[0] && !b[1]) * a[0]
         +(int)(!b[0] &&  b[1]) * a[1];
  
    // was any of the 2 values good?
    *good = b[0] || b[1];
  }
};


template<typename val_t>
void getChords( val_t chords[], int voxelIds[],
                int nChords,
                val_t ray[],
                val_t gridO[], val_t gridD[], int gridN[] )
{
  // Get params of first and last intersected planes
  bool  min_good,  max_good;
  val_t alpha_min, alpha_max;
  getAlphaMin(ray, gridO, gridD, gridN, &alpha_min, &min_good);
  getAlphaMax(ray, gridO, gridD, gridN, &alpha_max, &max_good);
  if(!(min_good && max_good)) return; // grid intersected at all?

  val_t length = length(ray);
  
  bool _sects[3];
  int idDimMin[3];
  int idDimMax[3];
  for(int dim=0; dim<3; dim++)
  {
    idDimMin[dim] = getIdDimMin(ray, gridO, gridD, gridN);
    idDimMax[dim] = getIdDimMax(ray, gridO, gridD, gridN);
  }

  // Get initial params for each dimension
  val_t alphaDim[3];
  for(int dim=0; dim<3; dim++)
  {
    alphaDim[dim] =
          (int)(ray[dim+3]> ray[dim]) * alphaFromId(idDimMin[dim], ray, gridO, gridD, gridN, dim)
        + (int)(ray[dim+3]<=ray[dim]) * alphaFromId(idDimMax[dim], ray, gridO, gridD, gridN, dim);
  }
  
  // Get index of first voxel crossed by the ray
  int id[3];
  for(int dim=0; dim<3; dim++)
  {
    id[dim] =
          std::floor(phiFromAlpha(
                (  .5 * min(alphaDim[0], alphaDim[1], alphaDim[2],
                         _sects[0],   _sects[1],   _sects[2])
                 + .5 * alpha_min),
                ray, gridO, gridD, gridN
          ));
  }
  
  // Initialize current param to first plane crossed
  val_t alpha_curr = alpha_min;

  // ITERATIONS
  int i = 0;
  while(alpha_curr < alpha_max)
  {
    for(int dim=0; dim<3; dim++)
    {
      int dimCrossed =
            (int)(alphaDim[dim] == min(alphaDim[0],alphaDim[1],alphaDim[2],
                                       _sects[0],  _sects[1],  _sects[2])
                  && alphaDim[dim]<=alpha_max);
      
      for(int writeDim=0; writeDim<3; writeDim++)
      {
        voxelIds[writeDim] += dimCrossed * id[writeDim];
        chords[writeDim]   += dimCrossed * (alphaDim[dim]-alpha_curr)*length;
      }
      
      alpha_curr = alphaDim[dim];
      updateAlpha(alphaDim[dim], ray, gridO, gridD, gridN, dim);
      updateId(   id[dim],       ray, gridO, gridD, gridN, dim);
      
      i++;
    }
  }
}


template<typename val_t>
int
getNChords(
      val_t ray[],
      val_t gridO[], val_t gridD[], int gridN[] )
{
//  if(!valid(ray, grid))
//  {
//    std::cerr << "ChordsCalc<...>" << std::endl
//              << "::calculateChordLengths(...) : "
//              << "Error: Ray_t starts/ends within the grid"
//              << std::endl;
//    throw 1;
//  }
  
  val_t   amin,     amax;
  bool    min_good, max_good;
  getAlphaMin(ray, gridO, gridD, gridN, &amin, &min_good);
  getAlphaMax(ray, gridO, gridD, gridN, &amax, &max_good);
  
//  if(min_good && max_good && amin<amax)
//  {
//    int N = 0;
//    for(int dim=0; dim<3; dim++)
//    {
//      N += getIdDimmax(ray, grid, dim);
//      N -= getIdDimmin(ray, grid, dim);
//      N += 1;
//    }
//    return N;
//  }
//  else
//    return 0;
  int N = 0;
  for(int dim=0; dim<3; dim++)
  {
    N += getIdDimmax(ray, gridO, gridD, gridN, dim);
    N -= getIdDimmin(ray, gridO, gridD, gridN, dim);
    N += 1;
  }

  return (int)(min_good && max_good) * N;
} 


template<typename val_t>
void
getCrossesPlanes(
      bool crosses[],
      val_t const ray[],
      val_t const gridO[], val_t const gridD[], int const gridN[] )
{
  for(int dim=0; dim<3; dim++)
    crosses[dim] = (ray[dim+3] - ray[dim] != 0);
}


//template<typename val_t>
//bool
//crossesAnyPlanes(
//      val_t ray[],
//      val_t gridO[], val_t gridD[], int gridN[] )
//{
//  bool crosses[3];
//  getCrossesPlanes(ray, gridO, gridD, gridN, crosses);
//  return crosses[0] || crosses[1] || crosses[2];
//}

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
 * @arg i
 * @arg ray
 * @arg gridO
 * @arg gridD
 * @arg ray       Array of ray start and end coordinates
 * @arg gridO     Array of grid origin coordinates
 * @arg gridD     Array of grid plane spacings for each axis
 * @arg gridN     Array of grid voxel numbers for each axis
 */
template<typename val_t>
val_t
alphaFromId(
      int const i,
      val_t const ray[],
      val_t const gridO[], val_t const gridD[], int const gridN[],
      int const dim )
{
  val_t divisor = ray[dim+3] - ray[dim];
  divisor += (divisor==0);

  return ( gridO[dim] + i * gridD[dim] - ray[dim] )
        /  divisor;
} 


template<typename val_t>
val_t
phiFromAlpha(
      val_t const alpha,
      val_t const ray[],
      val_t const gridO[], val_t const gridD[], int const gridN[],
      int const dim )
{
  return ( ray[dim] + alpha * (ray[dim+3] - ray[dim]) - gridO[dim] )
         /
         ( gridD[dim] );
} 


/**
 * @brief Function that for a given ray and a given grid determines for each
 *        dimension the minimum of all parameters of intersections of the ray
 *        with grid planes that are orthogonal to the dimension's coordinate
 *        axis.
 *
 * @tparam val_t  Coordinate type
 * @arg aDimmin   Result memory for mimimum intersection parameters
 * @arg ray       Array of ray start and end coordinates
 * @arg gridO     Array of grid origin coordinates
 * @arg gridD     Array of grid plane spacings for each axis
 * @arg gridN     Array of grid voxel numbers for each axis
 * @arg good      Array of minimum condition for each axis: true - perpendicular
 *                grid planes are intersected, false - not
 */
template<typename val_t>
void
getAlphaDimmin( 
      val_t aDimmin[],
      val_t const ray[],
      val_t const gridO[], val_t const gridD[], int const gridN[],
      bool const good[] )
{
  for(int dim=0; dim<3; dim++)
  {
    aDimmin[dim] = (int)good[dim] * MinFunctor<>()(
          alphaFromId(0,          ray, gridO, gridD, gridN, dim),
          alphaFromId(gridN[dim], ray, gridO, gridD, gridN, dim));
  }
} 


/**
 * @brief Function that for a given ray and a given grid determines for each
 *        dimension the maximum of all parameters of intersections of the ray
 *        with grid planes that are orthogonal to the dimension's coordinate
 *        axis.
 *
 * @tparam val_t  Coordinate type
 * @arg aDimmin   Result memory for maximum intersection parameters
 * @arg ray       Array of ray start and end coordinates
 * @arg gridO     Array of grid origin coordinates
 * @arg gridD     Array of grid plane spacings for each axis
 * @arg gridN     Array of grid voxel numbers for each axis
 * @arg good      Array of maximum condition for each axis: true - perpendicular
                  grid planes are intersected, false - not
 */
template<typename val_t>
void
getAlphaDimmax(
      val_t ray[],
      val_t gridO[], val_t gridD[], int gridN[],
      val_t aDimMax[], bool good[] )
{
  for(int dim=0; dim<3; dim++)
  {
     aDimMax[dim] = (int)good[dim] * MaxFunctor<>()(
          alphaFromId(0,          ray, gridO, gridD, gridN, dim),
          alphaFromId(gridN[dim], ray, gridO, gridD, gridN, dim)
                                                    );
  }
} 


template<typename val_t>
void
getAlphaMin(
      val_t ray[],
      val_t gridO[], val_t gridD[], int gridN[],
      val_t * aMin, bool * good )
{
  bool    crosses[3];
  val_t   adimmin[3];
  getAlphaDimmin(ray, gridO, gridD, gridN, adimmin, crosses);

  *good = true;
  for(int dim=0; dim<3; dim++)
  {
    val_t c    = ray[dim];
    val_t lim0 = gridO[dim];
    val_t lim1 = lim0 + gridN[dim]*gridD[dim];
    
//    if(!crosses[dim])
//    {
//      if(!( (c>lim0 && c<lim1) || (c<lim0 && c>lim1) ))
//        *good = false;
//    }
    *good = !(!crosses[dim] && !((c>lim0 && c<lim1) || (c<lim0 && c>lim1)));
  }

//  if(*good)
//    *amin = max(adimmin[0], adimmin[1], adimmin[2],
//                crosses[0], crosses[1], crosses[2]);
  *aMin = (int)(*good) * max(adimmin[0], adimmin[1], adimmin[2],
                             crosses[0], crosses[1], crosses[2]);
} 


template<typename val_t>
void
getAlphaMax(
      val_t ray[],
      val_t gridO[], val_t gridD[], int gridN[],
      val_t * aMax, bool * good )
{
//  bool    crosses[3];
//  Coord_t adimmax[3];
//  getAlphaDimmax(ray, grid, adimmax, crosses);
//
//  *good = true;
//  for(int dim=0; dim<3; dim++)
//  {
//    if(!crosses[dim])
//    {
//      Coord_t c    = ray.start()[dim];
//      Coord_t lim0 = grid.origin()[dim];
//      Coord_t lim1 = lim0 + grid.N()[dim]*grid.diff()[dim];
//      if(!( (c>lim0 && c<lim1) || (c<lim0 && c>lim1) ))
//        *good = false;
//    }
//  }
//
//  if(*good)
//    *amax = min(adimmax[0], adimmax[1], adimmax[2],
//                crosses[0], crosses[1], crosses[2]);
  bool    crosses[3];
  val_t   adimmax[3];
  getAlphaDimmax(ray, gridO, gridD, gridN, adimmax, crosses);

  *good = true;
  for(int dim=0; dim<3; dim++)
  {
    val_t c    = ray[dim];
    val_t lim0 = gridO[dim];
    val_t lim1 = lim0 + gridN[dim]*gridD[dim];
    
    *good = !(!crosses[dim] && !((c>lim0 && c<lim1) || (c<lim0 && c>lim1)));
  }

  *aMax = (int)(*good) * min(adimmax[0], adimmax[1], adimmax[2],
                             crosses[0], crosses[1], crosses[2]);
}


template<typename val_t>
void getIdFirst(
      val_t ray[],
      val_t gridO[], val_t gridD[], int gridN[],
      int idFirst[],
      bool good[])
{
  val_t   aMin;
  bool    aMinExists;
  getAlphaMin(ray, gridO, gridD, gridN, &aMin, &aMinExists);
  
  for(int dim=0; dim<3; dim++)
  {
    // from entry point, move inside first voxel then get voxel id
    idFirst[dim] = std::floor(phiFromAlpha(aMin + 0.5*aMinUp, ray, gridO, gridD, gridN, dim));
  }
}


    
//template<typename val_t>
//int
//getIdDimmin(
//      val_t ray[],
//      val_t gridO[], val_t gridD[], int gridN[],
//      int dim )
//{
//  val_t   alpha_min;
//  bool    min_good;
//  getAlphaMin(ray, gridO, gridD, gridN, &alpha_min, &min_good);
//  
//  val_t   alpha_dimmin[3];
//  bool    dimmin_good[3];
//  getAlphaDimmin(ray, gridO, gridD, gridN, alpha_dimmin, dimmin_good);
//
//
//  val_t   alpha_max;
//  bool    max_good;
//  getAlphaMax(ray, gridO, gridD, gridN, &alpha_max, &max_good);
//  
//  val_t   alpha_dimmax[3];
//  bool    dimmax_good[3];
//  getAlphaDimmax(ray, gridO, gridD, gridN, alpha_dimmax, dimmax_good);
//
////  if(ray[dim] < ray[dim+3])
////  {
////    if(alpha_dimmin[dim] != alpha_min)
////      return ceil(phiFromAlpha(alpha_min, ray, grid, dim));
////    else
////      return 1;
////  }
////  else
////  {
////    if(alpha_dimmax[dim] != alpha_max)
////      return ceil(phiFromAlpha(alpha_max, ray, grid, dim));
////    else
////      return 0;
////  }
//  return (int)((ray[dim] < ray[dim+3]) && (alpha_dimmin[dim] != alpha_min))
//            * ceil(phiFromAlpha(alpha_min, ray, gridO, gridD, gridN, dim))
//        +(int)((ray[dim] < ray[dim+3]) && (alpha_dimmin[dim] == alpha_min))
//            * 1
//      +
//         (int)((ray[dim] >= ray[dim+3]) && (alpha_dimmax[dim] != alpha_max))
//            * ceil(phiFromAlpha(alpha_max, ray, gridO, gridD, gridN, dim))
//        +(int)((ray[dim] >= ray[dim+3]) && (alpha_dimmax[dim] == alpha_max))
//            * 0;
//} 

    
//template<typename val_t>
//int
//getIdDimmax(
//      val_t ray[],
//      val_t gridO[], val_t gridD[], int gridN[],
//      int dim )
//{
//  val_t   alpha_max;
//  bool    max_good;
//  getAlphaMax(ray, gridO, gridD, gridN, &alpha_max, &max_good);
//
//  val_t   alpha_dimmax[3];
//  bool    dimmax_good[3];
//  getAlphaDimmax(ray, gridO, gridD, gridN, alpha_dimmax, dimmax_good);
//
//
//  val_t   alpha_min;
//  bool    min_good;
//  getAlphaMin(ray, gridO, gridD, gridN, &alpha_min, &min_good);
//
//  val_t   alpha_dimmin[3];
//  bool    dimmin_good[3];
//  getAlphaDimmin(ray, gridO, gridD, gridN, alpha_dimmin, dimmin_good);
//
////  if(ray.start()[dim] < ray.end()[dim])
////  {
////    if(alpha_dimmax[dim] != alpha_max)
////      return floor(phiFromAlpha(alpha_max, ray, grid, dim));
////    else
////      return grid.N()[dim];
////  }
////  else
////  {
////    if(alpha_dimmin[dim] != alpha_min)
////      return floor(phiFromAlpha(alpha_min, ray, grid, dim));
////    else
////      return grid.N()[dim]-1;
////  }
//  
//  return (int)(ray[dim] < ray[dim+3] && alpha_dimmax[dim] != alpha_max)
//            * floor(phiFromAlpha(alpha_max, ray, gridO, gridD, gridN, dim))
//        +(int)(ray[dim] < ray[dim+3] && alpha_dimmax[dim] == alpha_max)
//            * gridN[dim]
//      +
//         (int)(ray[dim] >=ray[dim+3] && alpha_dimmin[dim] != alpha_min)
//            * floor(phiFromAlpha(alpha_min, ray, gridO, gridD, gridN, dim))
//        +(int)(ray[dim] >=ray[dim+3] && alpha_dimmin[dim] == alpha_min)
//            * gridN[dim]-1;
//} 

    
//template<typename val_t>
//void
//updateAlpha(
//      val_t & alpha,
//      val_t ray[],
//      val_t gridO[], val_t gridD[], int gridN[],
//      int dim )
//{
//  alpha += gridD[dim] / std::abs(ray[dim+3] - ray[dim]);
//} 

    
//template<typename val_t>
//void
//updateId(
//      int & id,
//      val_t ray[],
//      val_t gridO[], val_t gridD[], int gridN[],
//      int dim )
//{
////  int i_update;
////  if(ray.start()[dim] < ray.end()[dim]){
////    i_update = 1;
////  } else {
////    i_update = -1;
////  }
////  id += i_update;
//  int i_update = (int)(ray[dim] < ray[dim+3]) * 1
//                +(int)(ray[dim] >=ray[dim+3]) *-1;
//  id += i_update;
//}


//template<typename Ray, typename Grid, typename Chord>
//bool      ChordsCalc<Ray, Grid, Chord>
//::valid( Ray_t ray, Grid_t grid )
//{
//#if ((defined DEBUG || defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
//  std::cout << "ChordsCalc<>::valid(Ray_t,Grid_t)"
//            << std::endl;
//#endif
//  bool      start_valid = false;
//  bool      end_valid   = false;
//  Coord_t   origin[3];
//  Coord_t   far[3];
//  
//  for(int dim=0; dim<3; dim++)
//  {
//    origin[dim] = grid.origin()[dim];
//    far[dim]    = grid.origin()[dim] + grid.N()[dim]*grid.diff()[dim];
//    
//    if(origin[dim] < far[dim])
//    {
//      if(ray.start()[dim] < origin[dim] || ray.start()[dim] > far[dim])
//        start_valid = true;
//      if(ray.end()[dim]   < origin[dim] || ray.end()[dim]   > far[dim])
//        end_valid = true;
//    }
//    else
//    {
//      if(ray.start()[dim] > origin[dim] || ray.start()[dim] < far[dim])
//        start_valid = true;
//      if(ray.end()[dim]   > origin[dim] || ray.end()[dim]   < far[dim])
//        end_valid = true;
//    }
//#if ((defined CHORDSCALC_DEBUG) && (NO_CHORDSCALC_DEBUG==0))
//    std::cout << "dim: " << dim
//              << "origin[dim]: " << origin[dim] << ", far[dim]: " << far[dim]
//              << "  start[dim]: " << ray.start()[dim]
//              << ", end[dim]: " << ray.end()[dim]
//              << std::endl;
//#endif
//  }
//  return start_valid && end_valid;
//}

#endif  // #ifndef CHORDSCALC_LOWLEVEL_HPP
