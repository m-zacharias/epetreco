#ifndef SIDDON_HPP
#define SIDDON_HPP

#include <cmath>
#include "Siddon_helper.hpp" // for special min
// #include "Ray.hpp"
// #include "Grid.hpp"
// #include "Vertex.hpp"

#ifdef DEBUG
#include <iostream>
#endif

#include <iostream> // debugging **********************



namespace Siddon{
  /* #########################################################################
   * ### Planes in direction x/y/z intersected by ray?
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  bool intersects__x( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  bool intersects__x( Ray ray, Grid grid )
  {
    return ray.end().x - ray.start().x != 0;
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  bool intersects__y( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  bool intersects__y( Ray ray, Grid grid )
  {
    return ray.end().y - ray.start().y != 0;
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  bool intersects__z( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  bool intersects__z( Ray ray, Grid grid )
  {
    return ray.end().z - ray.start().z != 0;
  }
 
  
  /* #########################################################################
   * ### Parameter of i'th intersection in direction x/y/z
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T alpha_from_i__x( int i, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t alpha_from_i__x( int i, Ray ray, Grid grid )
  {
      return (       grid.origin().x
               + i * grid.diff().x
               -      ray.start().x )
             /
             (        ray.end().x
               -      ray.start().x );
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T alpha_from_i__y( int i, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t alpha_from_i__y( int i, Ray ray, Grid grid )
  {
      return (       grid.origin().y
               + i * grid.diff().y
               -      ray.start().y )
             /
             (        ray.end().y
               -      ray.start().y );
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T alpha_from_i__z( int i, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t alpha_from_i__z( int i, Ray ray, Grid grid )
  {
      return (       grid.origin().z
               + i * grid.diff().z
               -      ray.start().z )
             /
             (        ray.end().z
               -      ray.start().z );
  }
 
  
  /* #########################################################################
   * ### Invert parameter to continuous (!) plane (!) "index" in direction x/y/z
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T phi_from_alpha__x( T alpha, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t phi_from_alpha__x( typename Ray::Vertex_t::Coord_t alpha, Ray ray, Grid grid )
  {
      return (   ray.start().x
               + alpha * (   ray.end().x
                           - ray.start().x )
               - grid.origin().x )
             /
             (   grid.diff().x );
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T phi_from_alpha__y( T alpha, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t phi_from_alpha__y( typename Ray::Vertex_t::Coord_t alpha, Ray ray, Grid grid )
  {
      return (   ray.start().y
               + alpha * (   ray.end().y
                           - ray.start().y )
               - grid.origin().y )
             /
             (   grid.diff().y );
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T phi_from_alpha__z( T alpha, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t phi_from_alpha__z( typename Ray::Vertex_t::Coord_t alpha, Ray ray, Grid grid )
  {
      return (   ray.start().z
               + alpha * (   ray.end().z
                           - ray.start().z )
               - grid.origin().z )
             /
             (   grid.diff().z );
  }


  /* #########################################################################
   * ### Get minimum intersection parameter of intersections with planes in
   * ### direction x/y/z
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T get_alpha_dimmin__x( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t get_alpha_dimmin__x( Ray ray, Grid grid )
  {
      return std::min(alpha_from_i__x<Ray,Grid>(0,         ray, grid),
                      alpha_from_i__x<Ray,Grid>(grid.Nx(), ray, grid));
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T get_alpha_dimmin__y( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t get_alpha_dimmin__y( Ray ray, Grid grid )
  {
      return std::min(alpha_from_i__y<Ray,Grid>(0,         ray, grid),
                      alpha_from_i__y<Ray,Grid>(grid.Ny(), ray, grid));
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T get_alpha_dimmin__z( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t get_alpha_dimmin__z( Ray ray, Grid grid )
  {
      return std::min(alpha_from_i__z<Ray,Grid>(0,         ray, grid),
                      alpha_from_i__z<Ray,Grid>(grid.Nz(), ray, grid));
  }
  
  
  /* #########################################################################
   * ### Get maximum intersection parameter of intersections with planes in
   * ### direction x/y/z
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T get_alpha_dimmax__x( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t get_alpha_dimmax__x( Ray ray, Grid grid )
  {
      return std::max(alpha_from_i__x<Ray,Grid>(0,         ray, grid),
                      alpha_from_i__x<Ray,Grid>(grid.Nx(), ray, grid));
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T get_alpha_dimmax__y( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t get_alpha_dimmax__y( Ray ray, Grid grid )
  {
      return std::max(alpha_from_i__y<Ray,Grid>(0,         ray, grid),
                      alpha_from_i__y<Ray,Grid>(grid.Ny(), ray, grid));
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T get_alpha_dimmax__z( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t get_alpha_dimmax__z( Ray ray, Grid grid )
  {
      return std::max(alpha_from_i__z<Ray,Grid>(0,         ray, grid),
                      alpha_from_i__z<Ray,Grid>(grid.Nz(), ray, grid));
  }
  

  /* #########################################################################
   * ### Does ray intersect planes in any direction at all?
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  bool alpha_min_exists( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  bool alpha_min_exists( Ray ray, Grid grid )
  {
      return    intersects__x<Ray,Grid>(ray, grid)
             || intersects__y<Ray,Grid>(ray, grid)
             || intersects__z<Ray,Grid>(ray, grid);
  }

//  coord_type get_alpha_min( coordVector const alpha_dimmin )
//  {
//      return max( max( alpha_dimmin[0],
//                       alpha_dimmin[1] ),
//                  alpha_dimmin[2] );
//  }

  
  /* #########################################################################
   * ### Get parameter of entry point, i.e. get minimum parameter of an
   * ### intersection of the ray with a plane that is adjacent to a voxel which
   * ### has a finite intersection length with that ray.
   * #########################################################################*/
  // Correct results only for `alpha_min_exists(...) == true`!!!
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T get_alpha_min( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t get_alpha_min( Ray ray, Grid grid )
  {
    typename Ray::Vertex_t::Coord_t temp;
    typename Ray::Vertex_t::Coord_t temp_min;
    bool       not_first = false;

    if(intersects__x<Ray,Grid>(ray, grid)) {
      temp = get_alpha_dimmin__x<Ray,Grid>(ray, grid);
      if(not_first) {
          temp_min = std::max(temp_min, temp); // !
      } else {
          temp_min = temp;
          not_first = true;
      }
    }
    if(intersects__y<Ray,Grid>(ray, grid)) {
      temp = get_alpha_dimmin__y<Ray,Grid>(ray, grid);
      if(not_first) {
          temp_min = std::max(temp_min, temp); // !
      } else {
          temp_min = temp;
          not_first = true;
      }
    }
    if(intersects__z<Ray,Grid>(ray, grid)) {
      temp = get_alpha_dimmin__z<Ray,Grid>(ray, grid);
      if(not_first) {
          temp_min = std::max(temp_min, temp); // !
      } else {
          temp_min = temp;
          not_first = true;
      }
    }
    if(!not_first) {
    std::cout << "Haha" << std::endl << std::flush; // debugging ***********************************************
      throw 1;
    }
    
    return temp_min;
  }
 
  
  /* #########################################################################
   * ### Does ray intersect planes in any direction at all?
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  bool alpha_max_exists( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  bool alpha_max_exists( Ray ray, Grid grid )
  {
      return    intersects__x<Ray,Grid>(ray, grid)
             || intersects__y<Ray,Grid>(ray, grid)
             || intersects__z<Ray,Grid>(ray, grid);
  }

// coord_type get_alpha_max( coordVector const alpha_dimmax )
// {
//     return min( min( alpha_dimmax[0],
//                      alpha_dimmax[1] ),
//                 alpha_dimmax[2] );
// }

  /* #########################################################################
   * ### Get parameter of exit point, i.e. get maximum parameter of an
   * ### intersection of the ray with a plane that is adjacent to a voxel which
   * ### has a finite intersection length with that ray
   * #########################################################################*/
  // Correct results only for `alpha_max_exists(...) == true`!!!
//  template<class T, template<class> class Ray, template<class> class Grid>
//  T get_alpha_max( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  typename Ray::Vertex_t::Coord_t get_alpha_max( Ray ray, Grid grid )
  {
      typename Ray::Vertex_t::Coord_t temp;
      typename Ray::Vertex_t::Coord_t temp_max;
      bool       not_first = false;
      
      if(intersects__x<Ray,Grid>(ray, grid)) {
        temp = get_alpha_dimmax__x<Ray,Grid>(ray, grid);
        if(not_first) {
          temp_max = std::min(temp_max, temp); // !
        }
        else {
          temp_max = temp;
          not_first = true;
        }
      }
      if(intersects__y<Ray,Grid>(ray, grid)) {
        temp = get_alpha_dimmax__y<Ray,Grid>(ray, grid);
        if(not_first) {
          temp_max = std::min(temp_max, temp); // !
        }
        else {
          temp_max = temp;
          not_first = true;
        }
      }
      if(intersects__z<Ray,Grid>(ray, grid)) {
        temp = get_alpha_dimmax__z<Ray,Grid>(ray, grid);
        if(not_first) {
          temp_max = std::min(temp_max, temp); // !
        }
        else {
          temp_max = temp;
          not_first = true;
        }
      }
      if(!not_first) {
        std::cout << "Haha 351" << std::endl << std::flush; // debugging *****************
        throw 1;
      }

      return temp_max;
  }
  

  /* #########################################################################
   * ### Get minimum of these two plane indices in direction x/y/z:
   * ### - of first intersected plane after the ray entered voxel space
   * ### - of last intersected plane including the outer plane of voxel space
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  int get_i_dimmin__x( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  int get_i_dimmin__x( Ray ray, Grid grid )
  {
      if(ray.start().x < ray.end().x) {
        typename Ray::Vertex_t::Coord_t alpha_min =    get_alpha_min<Ray,Grid>      (ray, grid);
        typename Ray::Vertex_t::Coord_t alpha_dimmin = get_alpha_dimmin__x<Ray,Grid>(ray, grid);
        
        if(alpha_dimmin != alpha_min) {
          return ceil(phi_from_alpha__x<Ray,Grid>(alpha_min, ray, grid));
        }
        else {
          return 1;
        }
      }
      else {
        typename Ray::Vertex_t::Coord_t alpha_max    = get_alpha_max<Ray,Grid>      (ray, grid);
        typename Ray::Vertex_t::Coord_t alpha_dimmax = get_alpha_dimmax__x<Ray,Grid>(ray, grid);
        
        if(alpha_dimmax != alpha_max) {
          return ceil(phi_from_alpha__x<Ray,Grid>(alpha_max, ray, grid));
        }
        else {
          return 0;
        }
      }
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  int get_i_dimmin__y( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  int get_i_dimmin__y( Ray ray, Grid grid )
  {
      if(ray.start().y < ray.end().y) {
        typename Ray::Vertex_t::Coord_t alpha_min =    get_alpha_min<Ray,Grid>      (ray, grid);
        typename Ray::Vertex_t::Coord_t alpha_dimmin = get_alpha_dimmin__y<Ray,Grid>(ray, grid);
        
        if(alpha_dimmin != alpha_min) {
          return ceil(phi_from_alpha__y<Ray,Grid>(alpha_min, ray, grid));
        }
        else {
          return 1;
        }
      }
      else {
        typename Ray::Vertex_t::Coord_t alpha_max    = get_alpha_max<Ray,Grid>      (ray, grid);
        typename Ray::Vertex_t::Coord_t alpha_dimmax = get_alpha_dimmax__y<Ray,Grid>(ray, grid);
        
        if(alpha_dimmax != alpha_max) {
          return ceil(phi_from_alpha__y<Ray,Grid>(alpha_max, ray, grid));
        }
        else {
          return 0;
        }
      }
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  int get_i_dimmin__z( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  int get_i_dimmin__z( Ray ray, Grid grid )
  {
      if(ray.start().z < ray.end().z) {
        typename Ray::Vertex_t::Coord_t alpha_min =    get_alpha_min<Ray,Grid>      (ray, grid);
        typename Ray::Vertex_t::Coord_t alpha_dimmin = get_alpha_dimmin__z<Ray,Grid>(ray, grid);
        
        if(alpha_dimmin != alpha_min) {
          return ceil(phi_from_alpha__z<Ray,Grid>(alpha_min, ray, grid));
        }
        else {
          return 1;
        }
      }
      else {
        typename Ray::Vertex_t::Coord_t alpha_max    = get_alpha_max<Ray,Grid>      (ray, grid);
        typename Ray::Vertex_t::Coord_t alpha_dimmax = get_alpha_dimmax__z<Ray,Grid>(ray, grid);
        
        if(alpha_dimmax != alpha_max) {
          return ceil(phi_from_alpha__z<Ray,Grid>(alpha_max, ray, grid));
        }
        else {
          return 0;
        }
      }
  }
 
  
  /* #########################################################################
   * ### Get maximum of these two plane indices in direction dim:
   * ###  - of first intersected plane after the ray entered voxel space
   * ###  - of last intersected plane including the outer plane of voxel space
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  int get_i_dimmax__x( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  int get_i_dimmax__x( Ray ray, Grid grid )
  {
    if(ray.start().x < ray.end().x) {
      typename Ray::Vertex_t::Coord_t alpha_max    = get_alpha_max<Ray,Grid>      (ray, grid);
      typename Ray::Vertex_t::Coord_t alpha_dimmax = get_alpha_dimmax__x<Ray,Grid>(ray, grid);
      
      if(alpha_dimmax != alpha_max) {
        return floor(phi_from_alpha__x<Ray,Grid>(alpha_max, ray, grid));
      }
      else {
        return grid.Nx();
      }
    }
    else {
      typename Ray::Vertex_t::Coord_t alpha_min    = get_alpha_min<Ray,Grid>      (ray, grid);
      typename Ray::Vertex_t::Coord_t alpha_dimmin = get_alpha_dimmin__x<Ray,Grid>(ray, grid);
      
      if(alpha_dimmin != alpha_min) {
        return floor(phi_from_alpha__x<Ray,Grid>(alpha_min, ray, grid));
      }
      else {
        return grid.Nx()-1;
      }
    }
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  int get_i_dimmax__y( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  int get_i_dimmax__y( Ray ray, Grid grid )
  {
    if(ray.start().y < ray.end().y) {
      typename Ray::Vertex_t::Coord_t alpha_max    = get_alpha_max<Ray,Grid>      (ray, grid);
      typename Ray::Vertex_t::Coord_t alpha_dimmax = get_alpha_dimmax__y<Ray,Grid>(ray, grid);
      
      if(alpha_dimmax != alpha_max) {
        return floor(phi_from_alpha__y<Ray,Grid>(alpha_max, ray, grid));
      }
      else {
        return grid.Ny();
      }
    }
    else {
      typename Ray::Vertex_t::Coord_t alpha_min    = get_alpha_min<Ray,Grid>      (ray, grid);
      typename Ray::Vertex_t::Coord_t alpha_dimmin = get_alpha_dimmin__y<Ray,Grid>(ray, grid);
      
      if(alpha_dimmin != alpha_min) {
        return floor(phi_from_alpha__y<Ray,Grid>(alpha_min, ray, grid));
      }
      else {
        return grid.Ny()-1;
      }
    }
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  int get_i_dimmax__z( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  int get_i_dimmax__z( Ray ray, Grid grid )
  {
    if(ray.start().z < ray.end().z) {
      typename Ray::Vertex_t::Coord_t alpha_max    = get_alpha_max<Ray,Grid>      (ray, grid);
      typename Ray::Vertex_t::Coord_t alpha_dimmax = get_alpha_dimmax__z<Ray,Grid>(ray, grid);
      
      if(alpha_dimmax != alpha_max) {
        return floor(phi_from_alpha__z<Ray,Grid>(alpha_max, ray, grid));
      }
      else {
        return grid.Nz();
      }
    }
    else {
      typename Ray::Vertex_t::Coord_t alpha_min    = get_alpha_min<Ray,Grid>      (ray, grid);
      typename Ray::Vertex_t::Coord_t alpha_dimmin = get_alpha_dimmin__z<Ray,Grid>(ray, grid);
      
      if(alpha_dimmin != alpha_min) {
        return floor(phi_from_alpha__z<Ray,Grid>(alpha_min, ray, grid));
      }
      else {
        return grid.Nz()-1;
      }
    }
  }
  

  /* #########################################################################
   * ### Get total number of grid planes crossed by the ray.  Actual number of
   * ### intersections might be smaller.  This is the case, if one or more
   * ### intersections cross more than one plane at the same point (i.e.
   * ### crossing at an edge or at a corner.)
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  int get_N_crossed_planes( Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  int get_N_crossed_planes( Ray ray, Grid grid )
  {
    int i_max[3] = {get_i_dimmax__x<Ray,Grid>(ray, grid),
                    get_i_dimmax__y<Ray,Grid>(ray, grid),
                    get_i_dimmax__z<Ray,Grid>(ray, grid)
                   };
    int i_min[3] = {get_i_dimmin__x<Ray,Grid>(ray, grid),
                    get_i_dimmin__y<Ray,Grid>(ray, grid),
                    get_i_dimmin__z<Ray,Grid>(ray, grid)
                   };
    return   i_max[0] - i_min[0]\
           + i_max[1] - i_min[1]\
           + i_max[2] - i_min[2]\
           + 3;
  }
 
  
  /* #########################################################################
   * ### Helper function for implementation of improved Siddon algorithm by
   * ### Jacobs et alii.
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  void update_alpha__x( T & alpha_x, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  void update_alpha__x( typename Ray::Vertex_t::Coord_t & alpha_x, Ray ray, Grid grid )
  {
    alpha_x += grid.diff().x / std::abs(ray.end().x - ray.start().x);
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  void update_alpha__x( T & alpha_y, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  void update_alpha__y( typename Ray::Vertex_t::Coord_t & alpha_y, Ray ray, Grid grid )
  {
    alpha_y += grid.diff().y / std::abs(ray.end().y - ray.start().y);
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  void update_alpha__x( T & alpha_z, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  void update_alpha__z( typename Ray::Vertex_t::Coord_t & alpha_z, Ray ray, Grid grid )
  {
    alpha_z += grid.diff().z / std::abs(ray.end().z - ray.start().z);
  }


  /* #########################################################################
   * ### Helper function for implementation of improved Siddon algorithm by
   * ### Jacobs et alii.
   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  void update_i__x( int & i_x, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  void update_i__x( int & i_x, Ray ray, Grid grid )
  {
    int i_update;
    if(ray.start().x < ray.end().x){
      i_update = 1;
    } else {
      i_update = -1;
    }
    i_x += i_update;
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  void update_i__y( int & i_y, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  void update_i__y( int & i_y, Ray ray, Grid grid )
  {
    int i_update;
    if(ray.start().y < ray.end().y){
      i_update = 1;
    } else {
      i_update = -1;
    }
    i_y += i_update;
  }
//  template<class T, template<class> class Ray, template<class> class Grid>
//  void update_i__z( int & i_z, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  void update_i__z( int & i_z, Ray ray, Grid grid )
  {
    int i_update;
    if(ray.start().z < ray.end().z){
      i_update = 1;
    } else {
      i_update = -1;
    }
    i_z += i_update;
  }

  
//  /* #########################################################################
//   * ### Implementation of the improved Siddon algorithm by Jacobs et alii.
//   * #########################################################################*/
//  template<class T, template<class> class Ray, template<class> class Grid>
//  void calculate_intersection_lengths( T * a, Ray<T> ray, Grid<T> grid )
//  {
//    T length =     ray.get_length();
//    T alpha_min =  get_alpha_min<T,Ray,Grid>(ray, grid);
//    T alpha_max =  get_alpha_max<T,Ray,Grid>(ray, grid);
//    
//    int i_x_min =  get_i_dimmin__x<T,Ray,Grid>(ray, grid);
//    int i_y_min =  get_i_dimmin__y<T,Ray,Grid>(ray, grid);
//    int i_z_min =  get_i_dimmin__z<T,Ray,Grid>(ray, grid);
//    
//    int i_x_max =  get_i_dimmax__x<T,Ray,Grid>(ray, grid);
//    int i_y_max =  get_i_dimmax__y<T,Ray,Grid>(ray, grid);
//    int i_z_max =  get_i_dimmax__z<T,Ray,Grid>(ray, grid);
//    
//    T alpha_x;
//    if(ray.end().x > ray.start().x) {
//      alpha_x =    alpha_from_i__x<T,Ray,Grid>(i_x_min, ray, grid);
//    }
//    else {
//      alpha_x =    alpha_from_i__x<T,Ray,Grid>(i_x_max, ray, grid);
//    }
//
//    T alpha_y;
//    if(ray.end().y > ray.start().y) {
//      alpha_y =    alpha_from_i__y<T,Ray,Grid>(i_y_min, ray, grid);
//    }
//    else {
//      alpha_y =    alpha_from_i__y<T,Ray,Grid>(i_y_max, ray, grid);
//    }
//
//    T alpha_z;
//    if(ray.end().z > ray.start().z) {
//      alpha_z =    alpha_from_i__z<T,Ray,Grid>(i_z_min, ray, grid);
//    }
//    else {
//      alpha_z =    alpha_from_i__z<T,Ray,Grid>(i_z_max, ray, grid);
//    }
//    
//    int i_x =      std::floor(phi_from_alpha__x<T,Ray,Grid>(((min(alpha_x, alpha_y, alpha_z)+alpha_min)/2.), ray, grid));
//    int i_y =      std::floor(phi_from_alpha__y<T,Ray,Grid>(((min(alpha_x, alpha_y, alpha_z)+alpha_min)/2.), ray, grid));
//    int i_z =      std::floor(phi_from_alpha__z<T,Ray,Grid>(((min(alpha_x, alpha_y, alpha_z)+alpha_min)/2.), ray, grid));
//    
//    T alpha_curr = alpha_min;
//
//#ifdef DEBUG
//    std::cout << "length:      " << length     << std::endl;
//    std::cout << "alpha_min :  " << alpha_min  << std::endl;
//    std::cout << "alpha_max :  " << alpha_max  << std::endl;
//    std::cout << std::endl;
//    std::cout << "i_x_min :    " << i_x_min    << std::endl;
//    std::cout << "i_y_min :    " << i_y_min    << std::endl;
//    std::cout << "i_z_min :    " << i_z_min    << std::endl;
//    std::cout << std::endl;
//    std::cout << "alpha_x :    " << alpha_x    << std::endl;
//    std::cout << "alpha_y :    " << alpha_y    << std::endl;
//    std::cout << "alpha_z :    " << alpha_z    << std::endl;
//    std::cout << "alpha_curr : " << alpha_curr << std::endl;
//    std::cout << std::endl;
//#endif
//
//    // Iterate
//    int i = 0;
//    while(alpha_curr < alpha_max){
//      if( alpha_x == min(alpha_x, alpha_y, alpha_z)) {
//#ifdef DEBUG
//        std::cout << "intersect x plane at alpha = " << alpha_x << std::endl;
//#endif
//        a[i] = (alpha_x - alpha_curr)*length;
//        update_i__x<T,Ray,Grid>(i_x, ray, grid);
//        alpha_curr = alpha_x;
//        update_alpha__x<T,Ray,Grid>(alpha_x, ray, grid);
//        
//        if(alpha_curr == alpha_y) {
//          update_i__y<T,Ray,Grid>(i_y, ray, grid);
//          update_alpha__y<T,Ray,Grid>(alpha_y, ray, grid);
//        }
//        if(alpha_curr == alpha_z) {
//          update_i__z<T,Ray,Grid>(i_z, ray, grid);
//          update_alpha__z<T,Ray,Grid>(alpha_z, ray, grid);
//        }
//      }
//      else if(alpha_y == min(alpha_x, alpha_y, alpha_z)) {
//#ifdef DEBUG
//        std::cout << "intersect y plane at alpha = " << alpha_y << std::endl;
//#endif
//        a[i] = (alpha_y - alpha_curr)*length;
//        update_i__y<T,Ray,Grid>(i_y, ray, grid);
//        alpha_curr = alpha_y;
//        update_alpha__y<T,Ray,Grid>(alpha_y, ray, grid);
//        
//        if(alpha_curr == alpha_z) {
//          update_i__z<T,Ray,Grid>(i_z, ray, grid);
//          update_alpha__z<T,Ray,Grid>(alpha_z, ray, grid);
//        }
//      }
//      else {
//#ifdef DEBUG
//        std::cout << "intersect z plane at alpha = " << alpha_z << std::endl;
//#endif
//        a[i] = (alpha_z - alpha_curr)*length;
//        update_i__z<T,Ray,Grid>(i_z, ray, grid);
//        alpha_curr = alpha_z;
//        update_alpha__z<T,Ray,Grid>(alpha_z, ray, grid);
//      }
//      i++;
//    }
//  }
  
  
  template<class T>
  struct Intersection
  {
    public:
      
      int idx, idy, idz;

      T length;

      Intersection() {}

      Intersection( int idx_, int idy_, int idz_, T length_ )
      : idx(idx_), idy(idy_), idz(idz_), length(length_) {}
  };
  
  
//  template<class T, template<class> class Ray, template<class> class Grid>
//  void calculate_intersection_lengths( Intersection<T> * a, Ray<T> ray, Grid<T> grid )
  template<class Ray, class Grid>
  void calculate_intersection_lengths( Intersection<typename Ray::Vertex_t::Coord_t> * a, Ray ray, Grid grid )
  {
    // #################################
    // INITIALISATION
    // #################################
#ifdef DEBUG
    std::cout << "calculate_intersection_lengths<> : start initialisation" << std::endl;
#endif
    bool _intersects__x = intersects__x<Ray, Grid>( ray, grid );                // any planes in x direction intersected?
    bool _intersects__y = intersects__y<Ray, Grid>( ray, grid );                // any planes in y direction intersected?
    bool _intersects__z = intersects__z<Ray, Grid>( ray, grid );                // any planes in z direction intersected?

    typename Ray::Vertex_t::Coord_t length =     ray.length();                         // length of ray (cartesian)
    typename Ray::Vertex_t::Coord_t alpha_min =  get_alpha_min<Ray,Grid>(ray, grid);   // param of first intersected plane
    typename Ray::Vertex_t::Coord_t alpha_max =  get_alpha_max<Ray,Grid>(ray, grid);   // param of last intersected plane
    
    int i_x_min =                         get_i_dimmin__x<Ray,Grid>(ray, grid); // min/max indices of planes - following that rather complicated case differentiation
    int i_y_min =                         get_i_dimmin__y<Ray,Grid>(ray, grid); // -||-
    int i_z_min =                         get_i_dimmin__z<Ray,Grid>(ray, grid); // -||-
    
    int i_x_max =                         get_i_dimmax__x<Ray,Grid>(ray, grid); // -||-
    int i_y_max =                         get_i_dimmax__y<Ray,Grid>(ray, grid); // -||-
    int i_z_max =                         get_i_dimmax__z<Ray,Grid>(ray, grid); // -||-

    
    // Get initial alpha params for each dimension
    typename Ray::Vertex_t::Coord_t alpha_x;
    if(ray.end().x > ray.start().x) {
      alpha_x =                           alpha_from_i__x<Ray,Grid>(i_x_min, ray, grid);
    }
    else {
      alpha_x =                           alpha_from_i__x<Ray,Grid>(i_x_max, ray, grid);
    }
    
    typename Ray::Vertex_t::Coord_t alpha_y;
    if(ray.end().y > ray.start().y) {
      alpha_y =                           alpha_from_i__y<Ray,Grid>(i_y_min, ray, grid);
    }
    else {
      alpha_y =                           alpha_from_i__y<Ray,Grid>(i_y_max, ray, grid);
    }
    
    typename Ray::Vertex_t::Coord_t alpha_z;
    if(ray.end().z > ray.start().z) {
      alpha_z =                           alpha_from_i__z<Ray,Grid>(i_z_min, ray, grid);
    }
    else {
      alpha_z =                           alpha_from_i__z<Ray,Grid>(i_z_max, ray, grid);
    }
    

    // Get index of first voxel crossed by the ray
    int i_x =                             std::floor(phi_from_alpha__x<Ray,Grid>(((min(alpha_x, alpha_y, alpha_z, _intersects__x, _intersects__y, _intersects__z)+alpha_min)/2.), ray, grid));
    int i_y =                             std::floor(phi_from_alpha__y<Ray,Grid>(((min(alpha_x, alpha_y, alpha_z, _intersects__x, _intersects__y, _intersects__z)+alpha_min)/2.), ray, grid));
    int i_z =                             std::floor(phi_from_alpha__z<Ray,Grid>(((min(alpha_x, alpha_y, alpha_z, _intersects__x, _intersects__y, _intersects__z)+alpha_min)/2.), ray, grid));
    

    // Initialise current position to the first plane crossed
    typename Ray::Vertex_t::Coord_t alpha_curr = alpha_min;

#ifdef DEBUG
    std::cout << "    length:      " << length     << std::endl;
    std::cout << "    alpha_min :  " << alpha_min  << std::endl;
    std::cout << "    alpha_max :  " << alpha_max  << std::endl;
    std::cout << std::endl;
    std::cout << "    i_x_min :    " << i_x_min    << std::endl;
    std::cout << "    i_y_min :    " << i_y_min    << std::endl;
    std::cout << "    i_z_min :    " << i_z_min    << std::endl;
    std::cout << std::endl;
    std::cout << "    i_x_max :    " << i_x_max    << std::endl;
    std::cout << "    i_y_max :    " << i_y_max    << std::endl;
    std::cout << "    i_z_max :    " << i_z_max    << std::endl;
    std::cout << std::endl;
    std::cout << "    alpha_x :    " << alpha_x    << std::endl;
    std::cout << "    alpha_y :    " << alpha_y    << std::endl;
    std::cout << "    alpha_z :    " << alpha_z    << std::endl;
    std::cout << "    alpha_curr : " << alpha_curr << std::endl;
    std::cout << std::endl;
    std::cout << "calculate_intersection_lengths<> : initialisation done" << std::endl;
#endif
    // #################################



    // #################################
    // ITERATIONS
    // #################################
#ifdef DEBUG
    std::cout << "calculate_intersection_lengths<> : start iterations" << std::endl;
#endif
    int i = 0;
//    while(i<1) {
    while(alpha_curr < alpha_max) {
#ifdef DEBUG
      std::cout << "    iteration " << i << std::endl;
#endif
      
      
      // X PLANE AHEAD
      if(     alpha_x == min(alpha_x, alpha_y, alpha_z, _intersects__x, _intersects__y, _intersects__z)) {
#ifdef DEBUG
        std::cout << "    intersect x plane at alpha = " << alpha_x << std::endl;
#endif
        a[i].idx    = i_x;                            // save current voxel x id
        a[i].idy    = i_y;                            // save current voxel y id
        a[i].idz    = i_z;                            // save current voxel z id
        a[i].length = (alpha_x - alpha_curr)*length;  // calc+save current intersection length

        alpha_curr = alpha_x;                         // update current position
        update_alpha__x<Ray,Grid>(alpha_x, ray, grid);// update "next x plane to be crossed"
        update_i__x<Ray,Grid>(i_x, ray, grid);        // update voxel x id
        
//        // If more than one plane are crossed at the same point:  Acknowledge
//        // only the first crossing, skip the others
//        if(alpha_curr == alpha_y)
//        {
//          update_i__y<Ray,Grid>(i_y, ray, grid);
//          update_alpha__y<Ray,Grid>(alpha_y, ray, grid);
//        }
//        if(alpha_curr == alpha_z)
//        {
//          update_i__z<Ray,Grid>(i_z, ray, grid);
//          update_alpha__z<Ray,Grid>(alpha_z, ray, grid);
//        }
//        // end skip
      }


      // Y PLANE AHEAD
      else if(alpha_y == min(alpha_x, alpha_y, alpha_z, _intersects__x, _intersects__y, _intersects__z)) {
#ifdef DEBUG
        std::cout << "    intersect y plane at alpha = " << alpha_y << std::endl;
#endif
        a[i].idx    = i_x;                            // save current voxel x id
        a[i].idy    = i_y;                            // save current voxel y id
        a[i].idz    = i_z;                            // save current voxel z id
        a[i].length = (alpha_y - alpha_curr)*length;  // calc+save current intersection length
        
        alpha_curr = alpha_y;                         // update current position
        update_alpha__y<Ray,Grid>(alpha_y, ray, grid);// update "next y plane to be crossed"
        update_i__y<Ray,Grid>(i_y, ray, grid);        // update voxel y id
        
//        // If more than one plane are crossed at the same point:  Acknowledge
//        // only the first crossing, skip the others
//        if(alpha_curr == alpha_z)
//        {
//          update_i__z<Ray,Grid>(i_z, ray, grid);
//          update_alpha__z<Ray,Grid>(alpha_z, ray, grid);
//        }
//        // skip end
      }


      // Z PLANE AHEAD
      else if(alpha_z == min(alpha_x, alpha_y, alpha_z, _intersects__x, _intersects__y, _intersects__z)) {
#ifdef DEBUG
        std::cout << "    intersect z plane at alpha = " << alpha_z << std::endl;
#endif
        a[i].idx    = i_x;
        a[i].idy    = i_y;
        a[i].idz    = i_z;
        a[i].length = (alpha_z - alpha_curr)*length;  // calc+save current intersection length

        alpha_curr = alpha_z;                         // update current position
        update_alpha__z<Ray,Grid>(alpha_z, ray, grid);// update "next z plane to be crossed"
        update_i__z<Ray,Grid>(i_z, ray, grid);        // update voxel z id
      }


      // None of the above => should not happen within "while" => error
      else {
        throw -1;
      }


#ifdef DEBUG
      std::cout << std::endl;
#endif
      i++;
    } 
#ifdef DEBUG
    std::cout << "calculate_intersection_lengths<> : iterations done" << std::endl;
#endif
  }
}

#endif  // #ifndef SIDDON_HPP
