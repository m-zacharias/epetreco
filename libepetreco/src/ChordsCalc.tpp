#include "ChordsCalc.hpp"
#include "Siddon_helper.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef DEBUG
#include <iostream>
#endif

template<typename Ray, typename Grid, typename Chord>
ChordsCalc<Ray, Grid, Chord>
::ChordsCalc()
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::ChordsCalc()"
            << std::endl;
#endif
#endif
}


template<typename Ray, typename Grid, typename Chord>
void      ChordsCalc<Ray, Grid, Chord>
::getChords( Chord_t * a, Ray_t ray, Grid_t grid )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::getChords(Chord_t*,Ray_t,Grid_t)"
            << std::endl;
#endif
#endif
  if(!valid(ray, grid))
  {
    std::cerr << "### ChordsCalc<...>::getChords(...) : "
              << "Error: Ray_t starts/ends within the grid"
              << std::endl;
    throw 1;
  }

  // #################################
  // INITIALISATION
  // #################################
#ifdef DEBUG
  std::cout << "### ChordsCalc<...>::getChords(...) : "
            << "start initialisation"
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
  std::cout << "### length:      " << length     << std::endl;
  std::cout << "### alpha_min :  " << alpha_min  << std::endl;
  std::cout << "### alpha_max :  " << alpha_max  << std::endl;
  std::cout << std::endl;
  std::cout << "### i_dim_min[0] :    " << i_dim_min[0]    << std::endl;
  std::cout << "### i_dim_min[1] :    " << i_dim_min[1]    << std::endl;
  std::cout << "### i_dim_min[2] :    " << i_dim_min[2]    << std::endl;
  std::cout << std::endl;
  std::cout << "### i_dim_max[0] :    " << i_dim_max[0]    << std::endl;
  std::cout << "### i_dim_max[1] :    " << i_dim_max[1]    << std::endl;
  std::cout << "### i_dim_max[2] :    " << i_dim_max[2]    << std::endl;
  std::cout << std::endl;
  std::cout << "### dimalpha[0] :    " << dimalpha[0]    << std::endl;
  std::cout << "### dimalpha[1] :    " << dimalpha[1]    << std::endl;
  std::cout << "### dimalpha[2] :    " << dimalpha[2]    << std::endl;
  std::cout << "### alpha_curr : " << alpha_curr << std::endl;
  std::cout << std::endl;
  std::cout << "### ChordsCalc<...>::getChords(...) : initialisation done"
            << std::endl;
#endif
  // #################################



  // #################################
  // ITERATIONS
  // #################################
#ifdef DEBUG
  std::cout << "### ChordsCalc<...>::getChords(...) : start iterations"
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
  std::cout << "### ChordsCalc<...>::getChords(...) : iterations done"
            << std::endl;
#endif
}


template<typename Ray, typename Grid, typename Chord>
bool      ChordsCalc<Ray, Grid, Chord>
::intersects( Ray_t ray, Grid_t grid, int dim )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::intersects(Ray_t,Grid_t,int)"
            << std::endl;
#endif
#endif
  return ray.end()[dim] - ray.start()[dim] != 0;
} 


template<typename Ray, typename Grid, typename Chord>
bool      ChordsCalc<Ray, Grid, Chord>
::intersectsAny( Ray_t ray, Grid_t grid )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::intersectsAny(Ray_t,Grid_t)"
            << std::endl;
#endif
#endif
  return   intersects(ray, grid, 0)\
        || intersects(ray, grid, 1)\
        || intersects(ray, grid, 2);
} 


template<typename Ray, typename Grid, typename Chord>
typename ChordsCalc<Ray, Grid, Chord>::Coord_t     ChordsCalc<Ray, Grid, Chord>
::alphaFromId( int i, Ray_t ray, Grid_t grid, int dim )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::alphaFromId(int,Ray_t,Grid_t,int)"
            << std::endl;
#endif
#endif
  return (       grid.origin()[dim]
           + i * grid.diff()[dim]
           -      ray.start()[dim] )
         /
         (        ray.end()[dim]
           -      ray.start()[dim] );
} 


template<typename Ray, typename Grid, typename Chord>
typename ChordsCalc<Ray, Grid, Chord>::Coord_t     ChordsCalc<Ray, Grid, Chord>
::phiFromAlpha(
      Coord_t alpha, Ray_t ray, Grid_t grid, int dim )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::phiFromAlpha(Coord_t,Ray_t,Grid_t,int)"
            << std::endl;
#endif
#endif
  return (   ray.start()[dim]
           + alpha * (   ray.end()[dim]
                       - ray.start()[dim] )
           - grid.origin()[dim] )
         /
         (   grid.diff()[dim] );
} 


template<typename Ray, typename Grid, typename Chord>
typename ChordsCalc<Ray, Grid, Chord>::Coord_t     ChordsCalc<Ray, Grid, Chord>
::getAlphaDimmin( Ray_t ray, Grid_t grid, int dim )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::getAlphaDimmin(Ray_t,Grid_t,int)"
            << std::endl;
#endif
#endif
  return std::min(alphaFromId(0,             ray, grid, dim),
                  alphaFromId(grid.N()[dim], ray, grid, dim));
} 


template<typename Ray, typename Grid, typename Chord>
typename ChordsCalc<Ray, Grid, Chord>::Coord_t     ChordsCalc<Ray, Grid, Chord>
::getAlphaDimmax( Ray_t ray, Grid_t grid, int dim )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::getAlphaDimmax(Ray_t,Grid_t,int)"
            << std::endl;
#endif
#endif
  return std::max(alphaFromId(0,             ray, grid, dim),
                  alphaFromId(grid.N()[dim], ray, grid, dim));
} 


template<typename Ray, typename Grid, typename Chord>
typename ChordsCalc<Ray, Grid, Chord>::Coord_t     ChordsCalc<Ray, Grid, Chord>
::getAlphaMin( Ray_t ray, Grid_t grid )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::getAlphaMin(Ray_t,Grid_t)"
            << std::endl;
#endif
#endif
  typename Ray_t::Vertex_t::Coord_t temp;
  typename Ray_t::Vertex_t::Coord_t temp_min;
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
    throw 1;
  }
  
  return temp_min;
} 


template<typename Ray, typename Grid, typename Chord>
typename ChordsCalc<Ray, Grid, Chord>::Coord_t     ChordsCalc<Ray, Grid, Chord>
::getAlphaMax( Ray_t ray, Grid_t grid )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::getAlphaMax(Ray_t,Grid_t)"
            << std::endl;
#endif
#endif
  Coord_t temp;
  Coord_t temp_max;
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
    throw 1;
  }
  
  return temp_max;
} 

    
template<typename Ray, typename Grid, typename Chord>
int     ChordsCalc<Ray, Grid, Chord>
::getIdDimmin( Ray_t ray, Grid_t grid, int dim )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::getIdDimmin(Ray_t,Grid_t,int)"
            << std::endl;
#endif
#endif
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

    
template<typename Ray, typename Grid, typename Chord>
int     ChordsCalc<Ray, Grid, Chord>
::getIdDimmax( Ray_t ray, Grid_t grid, int dim )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::GetIdDimmax(Ray_t,Grid_t,int)"
            << std::endl;
#endif
#endif
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

#include <iostream> 
template<typename Ray, typename Grid, typename Chord>
int     ChordsCalc<Ray, Grid, Chord>
::getNChords( Ray_t ray, Grid_t grid )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::getNChords(Ray_t,Grid_t)"
            << std::endl;
#endif
#endif
  if(!valid(ray, grid))
  {
    std::cerr << "ChordsCalc<...>" << std::endl
              << "::calculateChordLengths(...) : "
              << "Error: Ray_t starts/ends within the grid"
              << std::endl;
    throw 1;
  }

  int N = 0;
  for(int dim=0; dim<3; dim++)
  {
    N += getIdDimmax(ray, grid, dim);
    N -= getIdDimmin(ray, grid, dim);
    N += 1;
  }
  return N;
} 

    
template<typename Ray, typename Grid, typename Chord>
void      ChordsCalc<Ray, Grid, Chord>
::updateAlpha(
      Coord_t & alpha, Ray_t ray, Grid_t grid,
      int dim )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::updateAlpha(Coord_t&,Ray_t,Grid_t,int)"
            << std::endl;
#endif
#endif
  alpha += grid.diff()[dim] / std::abs(ray.end()[dim] - ray.start()[dim]);
} 

    
template<typename Ray, typename Grid, typename Chord>
void      ChordsCalc<Ray, Grid, Chord>
::updateId( int & id, Ray_t ray, Grid_t grid, int dim )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::updateId(int&,Ray_t,Grid_t,int)"
            << std::endl;
#endif
#endif
  int i_update;
  if(ray.start()[dim] < ray.end()[dim]){
    i_update = 1;
  } else {
    i_update = -1;
  }
  id += i_update;
}


template<typename Ray, typename Grid, typename Chord>
bool      ChordsCalc<Ray, Grid, Chord>
::valid( Ray_t ray, Grid_t grid )
{
#ifdef DEBUG
#ifndef NO_CHORDSCALC_DEBUG
  std::cout << "ChordsCalc<>::valid(Ray_t,Grid_t)"
            << std::endl;
#endif
#endif
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

