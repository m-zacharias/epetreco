#include "IntersectionLengthFunctor.hpp"
#include "Siddon_helper.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>


template<typename Ray, typename Grid, typename Intersection>
IntersectionLengthFunctor<Ray, Grid, Intersection>
::IntersectionLengthFunctor() {}


template<typename Ray, typename Grid, typename Intersection>
void      IntersectionLengthFunctor<Ray, Grid, Intersection>
::calculateIntersectionLengths( Intersection * a, Ray ray, Grid grid )
{
  if(!valid(ray, grid))
  {
    std::cerr << "IntersectionLengthFunctor<...>" << std::endl
              << "::calculateIntersectionLengths(...) : "
              << "Error: Ray starts/ends within the grid"
              << std::endl;
    throw 1;
  }

  // #################################
  // INITIALISATION
  // #################################
#ifdef DEBUG
  std::cout << "IntersectionLengthFunctor<...>" << std::endl
            << "::calculateIntersectionLengths(...) : start initialisation"
            << std::endl;
#endif
  Coord_t length      = ray.length();             // length of ray (cartesian)
  Coord_t alpha_min   = getAlphaMin(ray, grid);   // param of first intersected plane
  Coord_t alpha_max   = getAlphaMax(ray, grid);   // param of last intersected plane

  bool _intersects[3];
  int  i_dim_min[3];
  int  i_dim_max[3];
  for(int dim=0; dim<3; dim++)
  {
    _intersects[dim]  = intersects(ray, grid, dim);// any planes intersected in dim?
    i_dim_min[dim]    = getIdDimmin(ray, grid, dim);
    i_dim_max[dim]    = getIdDimmax(ray, grid, dim);
  }

  // Get initial alpha params for each dimension
  Coord_t dimalpha[3];
  for(int dim=0; dim<3; dim++)
  {
    if(ray.end()[dim] > ray.start()[dim]) {
      dimalpha[dim]   = alphaFromId(i_dim_min[dim], ray, grid, dim);
    } else {
      dimalpha[dim]   = alphaFromId(i_dim_max[dim], ray, grid, dim);
    }
  }

  // Get index of first voxel crossed by the ray
  int id[3];
  for(int dim=0; dim<3; dim++)
  {
    id[dim] = std::floor(
                phiFromAlpha(
                  ( ( min(dimalpha[0],    dimalpha[1],    dimalpha[2],
                          _intersects[0], _intersects[1], _intersects[2]
                         ) + alpha_min
                    ) * 0.5
                  ), ray, grid, dim
                )
              );
  }
  
  // Initialise current position to the first plane crossed
  Coord_t alpha_curr = alpha_min;

#ifdef DEBUG
  std::cout << "    length:      " << length     << std::endl;
  std::cout << "    alpha_min :  " << alpha_min  << std::endl;
  std::cout << "    alpha_max :  " << alpha_max  << std::endl;
  std::cout << std::endl;
  std::cout << "    i_dim_min[0] :    " << i_dim_min[0]    << std::endl;
  std::cout << "    i_dim_min[1] :    " << i_dim_min[1]    << std::endl;
  std::cout << "    i_dim_min[2] :    " << i_dim_min[2]    << std::endl;
  std::cout << std::endl;
  std::cout << "    i_dim_max[0] :    " << i_dim_max[0]    << std::endl;
  std::cout << "    i_dim_max[1] :    " << i_dim_max[1]    << std::endl;
  std::cout << "    i_dim_max[2] :    " << i_dim_max[2]    << std::endl;
  std::cout << std::endl;
  std::cout << "    dimalpha[0] :    " << dimalpha[0]    << std::endl;
  std::cout << "    dimalpha[1] :    " << dimalpha[1]    << std::endl;
  std::cout << "    dimalpha[2] :    " << dimalpha[2]    << std::endl;
  std::cout << "    alpha_curr : " << alpha_curr << std::endl;
  std::cout << std::endl;
  std::cout << "IntersectionLengthFunctor<...>" << std::endl
            << "::calculateIntersectionLengths(...) : initialisation done"
            << std::endl;
#endif
  // #################################



  // #################################
  // ITERATIONS
  // #################################
#ifdef DEBUG
  std::cout << "IntersectionLengthFunctor<...>" << std::endl
            << "::calculateIntersectionLengths(...) : start iterations"
            << std::endl;
#endif
  int i = 0;
//  while(i<1) {
  while(alpha_curr < alpha_max)
  {
    bool no_crossing = true;
    for(int dim=0; dim<3; dim++)
    {
      if(dimalpha[dim] == min(dimalpha[0],    dimalpha[1],    dimalpha[2],
                              _intersects[0], _intersects[1], _intersects[2])
         && dimalpha[dim]<=alpha_max)
      {
#ifdef DEBUG
        std::cout << "    iteration " << i << std::endl;
        std::cout << "    intersect [" << dim << "] plane at alpha = "
                  << dimalpha[dim]
                  << std::endl;
        std::cout << "    id: (" 
                  << id[0] << "," << id[1] << "," << id[2] << "), "
                  << "length: "
                  << (dimalpha[dim]-alpha_curr)*length
                  << std::endl << std::endl;
#endif
        // Save current voxel id, intersection length
        a[i].setId    (id);
        a[i].setLength((dimalpha[dim] - alpha_curr)*length);
        
        // Update running parameters
        alpha_curr  = dimalpha[dim];
        updateAlpha(dimalpha[dim], ray, grid, dim);
        updateId   (id[dim],       ray, grid, dim);
        
        no_crossing = false;
        i++;
      }
    }
    // No crossing in any dimension => should not happen within "while" => error
    if(no_crossing)
      throw -1;
  } 
#ifdef DEBUG
  std::cout << "IntersectionLengthFunctor<...>" << std::endl
            << "::calculateIntersectionLengths(...) : iterations done"
            << std::endl;
#endif
}


template<typename Ray, typename Grid, typename Intersection>
bool      IntersectionLengthFunctor<Ray, Grid, Intersection>
::intersects( Ray ray, Grid grid, int dim )
{
  return ray.end()[dim] - ray.start()[dim] != 0;
} 


template<typename Ray, typename Grid, typename Intersection>
bool      IntersectionLengthFunctor<Ray, Grid, Intersection>
::intersectsAny( Ray ray, Grid grid )
{
  return   intersects(ray, grid, 0)\
        || intersects(ray, grid, 1)\
        || intersects(ray, grid, 2);
} 


template<typename Ray, typename Grid, typename Intersection>
typename Ray::Vertex_t::Coord_t     IntersectionLengthFunctor<Ray, Grid, Intersection>
::alphaFromId( int i, Ray ray, Grid grid, int dim )
{
  return (       grid.origin()[dim]
           + i * grid.diff()[dim]
           -      ray.start()[dim] )
         /
         (        ray.end()[dim]
           -      ray.start()[dim] );
} 


template<typename Ray, typename Grid, typename Intersection>
typename Ray::Vertex_t::Coord_t     IntersectionLengthFunctor<Ray, Grid, Intersection>
::phiFromAlpha(
      typename Ray::Vertex_t::Coord_t alpha, Ray ray, Grid grid, int dim )
{
  return (   ray.start()[dim]
           + alpha * (   ray.end()[dim]
                       - ray.start()[dim] )
           - grid.origin()[dim] )
         /
         (   grid.diff()[dim] );
} 


template<typename Ray, typename Grid, typename Intersection>
typename Ray::Vertex_t::Coord_t     IntersectionLengthFunctor<Ray, Grid, Intersection>
::getAlphaDimmin( Ray ray, Grid grid, int dim )
{
  return std::min(alphaFromId(0,             ray, grid, dim),
                  alphaFromId(grid.N()[dim], ray, grid, dim));
} 


template<typename Ray, typename Grid, typename Intersection>
typename Ray::Vertex_t::Coord_t     IntersectionLengthFunctor<Ray, Grid, Intersection>
::getAlphaDimmax( Ray ray, Grid grid, int dim )
{
  return std::max(alphaFromId(0,             ray, grid, dim),
                  alphaFromId(grid.N()[dim], ray, grid, dim));
} 


template<typename Ray, typename Grid, typename Intersection>
typename Ray::Vertex_t::Coord_t     IntersectionLengthFunctor<Ray, Grid, Intersection>
::getAlphaMin( Ray ray, Grid grid )
{
  typename Ray::Vertex_t::Coord_t temp;
  typename Ray::Vertex_t::Coord_t temp_min;
  bool       not_first = false;

  for(int dim=0; dim<3; dim++)
  {
    if(intersects(ray, grid, dim)) {
      temp = getAlphaDimmin(ray, grid, dim);
      if(not_first) {
        temp_min = std::max(temp_min, temp);
      } else {
        temp_min = temp;
        not_first = true;
      }
    }
  }

  if(!not_first) {
#ifdef DEBUG
    std::cout << "Haha" << std::endl << std::flush; // debugging *****************
#endif
    throw 1;
  }
  
  return temp_min;
} 


template<typename Ray, typename Grid, typename Intersection>
typename Ray::Vertex_t::Coord_t     IntersectionLengthFunctor<Ray, Grid, Intersection>
::getAlphaMax( Ray ray, Grid grid )
{
  typename Ray::Vertex_t::Coord_t temp;
  typename Ray::Vertex_t::Coord_t temp_max;
  bool       not_first = false;

  for(int dim=0; dim<3; dim++)
  {
    if(intersects(ray, grid, dim)) {
      temp = getAlphaDimmax(ray, grid, dim);
      if(not_first) {
        temp_max = std::min(temp_max, temp);
      } else {
        temp_max = temp;
        not_first = true;
      }
    }
  }

  if(!not_first) {
#ifdef DEBUG
    std::cout << "Haha" << std::endl << std::flush; // debugging *****************
#endif
    throw 1;
  }
  
  return temp_max;
} 

    
template<typename Ray, typename Grid, typename Intersection>
int     IntersectionLengthFunctor<Ray, Grid, Intersection>
::getIdDimmin( Ray ray, Grid grid, int dim )
{
  if(ray.start()[dim] < ray.end()[dim]) {
    Coord_t alpha_min =    getAlphaMin   (ray, grid);
    Coord_t alpha_dimmin = getAlphaDimmin(ray, grid, dim);
    
    if(alpha_dimmin != alpha_min) {
      return ceil(phiFromAlpha(alpha_min, ray, grid, dim));
    }
    else {
      return 1;
    }
  }
  else {
    Coord_t alpha_max    = getAlphaMax   (ray, grid);
    Coord_t alpha_dimmax = getAlphaDimmax(ray, grid, dim);
    
    if(alpha_dimmax != alpha_max) {
      return ceil(phiFromAlpha(alpha_max, ray, grid, dim));
    }
    else {
      return 0;
    }
  }
} 

    
template<typename Ray, typename Grid, typename Intersection>
int     IntersectionLengthFunctor<Ray, Grid, Intersection>
::getIdDimmax( Ray ray, Grid grid, int dim )
{
  if(ray.start()[dim] < ray.end()[dim]) {
    Coord_t alpha_max    = getAlphaMax   (ray, grid);
    Coord_t alpha_dimmax = getAlphaDimmax(ray, grid, dim);
    
    if(alpha_dimmax != alpha_max) {
      return floor(phiFromAlpha(alpha_max, ray, grid, dim));
    }
    else {
      return grid.N()[dim];
    }
  }
  else {
    Coord_t alpha_min    = getAlphaMin   (ray, grid);
    Coord_t alpha_dimmin = getAlphaDimmin(ray, grid, dim);
    
    if(alpha_dimmin != alpha_min) {
      return floor(phiFromAlpha(alpha_min, ray, grid, dim));
    }
    else {
      return grid.N()[dim]-1;
    }
  }
} 

 
template<typename Ray, typename Grid, typename Intersection>
int     IntersectionLengthFunctor<Ray, Grid, Intersection>
::getNCrossedVoxels( Ray ray, Grid grid )
{
  if(!valid(ray, grid))
  {
    std::cerr << "IntersectionLengthFunctor<...>" << std::endl
              << "::calculateIntersectionLengths(...) : "
              << "Error: Ray starts/ends within the grid"
              << std::endl;
    throw 1;
  }

  int N = 0;
  for(int dim=0; dim<3; dim++)
  {
#ifdef DEBUG_ILF
    std::cout << "dim: " << dim
              << ", IdDimmax: " << getIdDimmax(ray, grid, dim)
              << ", IdDimmin: " << getIdDimmin(ray, grid, dim)
              << std::endl;
#endif
    N += getIdDimmax(ray, grid, dim);
    N -= getIdDimmin(ray, grid, dim);
    N += 1;
  }
  return N;
} 

    
template<typename Ray, typename Grid, typename Intersection>
void      IntersectionLengthFunctor<Ray, Grid, Intersection>
::updateAlpha(
      typename Ray::Vertex_t::Coord_t & alpha, Ray ray, Grid grid,
      int dim )
{
  alpha += grid.diff()[dim] / std::abs(ray.end()[dim] - ray.start()[dim]);
} 

    
template<typename Ray, typename Grid, typename Intersection>
void      IntersectionLengthFunctor<Ray, Grid, Intersection>
::updateId( int & id, Ray ray, Grid grid, int dim )
{
  int i_update;
  if(ray.start()[dim] < ray.end()[dim]){
    i_update = 1;
  } else {
    i_update = -1;
  }
  id += i_update;
}


template<typename Ray, typename Grid, typename Intersection>
bool      IntersectionLengthFunctor<Ray, Grid, Intersection>
::valid( Ray ray, Grid grid )
{
  bool      start_valid = false;
  bool      end_valid   = false;
  Coord_t   origin[3];
  Coord_t   far[3];
  
  for(int dim=0; dim<3; dim++)
  {
    origin[dim] = grid.origin()[dim];
    far[dim]    = grid.origin()[dim] + grid.N()[dim]*grid.diff()[dim];
    
    if(origin[dim] < far[dim])
    {
      if(ray.start()[dim] < origin[dim] || ray.start()[dim] > far[dim])
        start_valid = true;
      if(ray.end()[dim] < origin[dim] || ray.end()[dim] > far[dim])
        end_valid = true;
    } else
    {
      if(ray.start()[dim] > origin[dim] || ray.start()[dim] < far[dim])
        start_valid = true;
      if(ray.end()[dim] > origin[dim] || ray.end()[dim] < far[dim])
        end_valid = true;
    }
  }

  return start_valid && end_valid;
}

