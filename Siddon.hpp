#ifndef SIDDON_HPP
#define SIDDON_HPP

#include <cmath>
#include <algorithm> // for min, max
//#include "CUDA_HandleError.hpp"
#ifdef DEBUG
#include <iostream>
#endif

typedef float coord_type;

template<typename T>
struct Vector {
    T x_,
      y_,
      z_;

    Vector( T const x,
            T const y,
            T const z )
    : x_(x), y_(y), z_(z) {}

    Vector( Vector const & ori )
    : x_(ori.x_), y_(ori.y_), z_(ori.z_) {}

    T & operator[]( int i ) {
        if( i%3 == 0 ) {
            return x_;
        }
        else if( i%3 ==1 ) {
            return y_;
        }
        else {
            return z_;
        }
    }
};

template<typename T>
struct Ray {
    Vector<T>   start_,
                end_;

    Ray( Vector<T> const start,
         Vector<T> const end )
    : start_(start), end_(end) {}

    Ray( Ray const & ori )
    : start_(ori.start_), end_(ori.end_) {}

    Vector<T> get_start( void ) const
    {
        return start_;
    }

    Vector<T> get_end( void ) const
    {
        return end_;
    }

    T get_length( void ) const
    {
      Vector<T> start = get_start();
      Vector<T> end   = get_end();
      return std::sqrt(  (end[0]-start[0])*(end[0]-start[0])\
                       + (end[1]-start[1])*(end[1]-start[1])\
                       + (end[2]-start[2])*(end[2]-start[2])\
                      );
    }
};

//! Class template
template<typename T>
struct Grid {
    Vector<T>   origin_,
                difference_;
    Vector<int> N_;

    //! Constructor
    Grid( Vector<T> const origin,
          Vector<T> const difference,
          Vector<int> const N )
    : origin_(origin), difference_(difference), N_(N) {}
    
    //! Copy constructor
    Grid( Grid const & ori )
    : origin_(ori.origin_), difference_(ori.difference_), N_(ori.N_) {}

    Vector<T> get_origin( void ) const
    {
        return origin_;
    }

    Vector<T> get_difference( void ) const
    {
        return difference_;
    }

    Vector<int> get_N( void ) const
    {
        return N_;
    }
};

template<typename T>
T min( T a, T b, T c )
{
  return std::min(std::min(a, b), c);
}

namespace Siddon {
    typedef Vector<coord_type>  coordVector;
    typedef Ray<coord_type>     coordRay;
    typedef Grid<coord_type>    coordGrid;
    
    /* Planes in direction dim intersected by ray? */
    bool intersects( int dim, coordRay ray, coordGrid grid )
    {
        return ray.get_end()[dim] - ray.get_start()[dim] != 0;
    }
    
    /* Parameter of i'th intersection in direction dim */
    coord_type alpha_from_i( int i, int dim, coordRay ray, coordGrid grid )
    {
        return (       grid.get_origin()[dim]
                 + i * grid.get_difference()[dim]
                 -      ray.get_start()[dim] )
               /
               (        ray.get_end()[dim]
                 -      ray.get_start()[dim] );
    }
    
    /* Invert parameter to plane (!) index in direction dim */
    coord_type phi_from_alpha( coord_type alpha, int dim, coordRay ray,
                               coordGrid grid )
    {
        return (    ray.get_start()[dim]
                 + alpha * (   ray.get_end()[dim]
                             - ray.get_start()[dim] )
                 - grid.get_origin()[dim] )
               /
               (   grid.get_difference()[dim] );
    }
    
//    coordVector get_alpha_dimmin( coordRay ray, coordGrid grid )
//    {
//        return coordVector( min( alpha_from_i(0,0,ray,grid),
//                                 alpha_from_i(grid.get_N()[0],0,ray,grid) ),
//
//                            min( alpha_from_i(0,1,ray,grid),
//                                 alpha_from_i(grid.get_N()[1],1,ray,grid) ),
//     
//                            min( alpha_from_i(0,2,ray,grid),
//                                 alpha_from_i(grid.get_N()[2],2,ray,grid) ) );
//    }

    /* Get minimum intersection parameter of intersections with planes in
     * direction dim */
    coord_type get_alpha_dimmin( int dim, coordRay ray, coordGrid grid )
    {
        return std::min(alpha_from_i(0,                dim,ray,grid),
                        alpha_from_i(grid.get_N()[dim],dim,ray,grid));
    }
    
//    coordVector get_alpha_dimmax( coordRay ray, coordGrid grid )
//    {
//        return Vector( max( alpha_from_i(0,0,ray,grid),
//                            alpha_from_i(grid.get_N()[0],0,ray,grid) ),
//
//                       max( alpha_from_i(0,1,ray,grid),
//                            alpha_from_i(grid.get_N()[1],1,ray,grid) ),
//
//                       max( alpha_from_i(0,2,ray,grid),
//                            alpha_from_i(grid.get_N()[2],2,ray,grid) ) );
//    }
    
    /* Get maximum intersection parameter of intersections with planes in
     * direction dim */
    coord_type get_alpha_dimmax( int dim, coordRay ray, coordGrid grid )
    {
        return std::max(alpha_from_i(0,                dim,ray,grid),
                        alpha_from_i(grid.get_N()[dim],dim,ray,grid));
    }
    
    /* Does ray intersect planes in any direction at all? */
    bool alpha_min_exists( coordRay ray, coordGrid grid )
    {
        return    intersects(0,ray,grid)
               || intersects(1,ray,grid)
               || intersects(2,ray,grid);
    }

//    coord_type get_alpha_min( coordVector const alpha_dimmin )
//    {
//        return max( max( alpha_dimmin[0],
//                         alpha_dimmin[1] ),
//                    alpha_dimmin[2] );
//    }

    
    /* Get parameter of entry point, i.e. get minimum parameter of an
     * intersection of the ray with a plane that is adjacent to a voxel which
     * has a finite intersection length with that ray. */
    // Correct results only for `alpha_min_exists(...) == true`!!!
    coord_type get_alpha_min( coordRay ray, coordGrid grid )
    {
        coord_type temp;
        coord_type temp_min;
        bool       not_first = false;
        for(int dim=0;dim<3;dim++)
        {
            if(intersects(dim,ray,grid))
            {
                temp = get_alpha_dimmin(dim,ray,grid);
                if(not_first)
                {
                    temp_min = std::max(temp_min,temp); // !
                } else
                {
                    temp_min = temp;
                    not_first = true;
                }
            }
        }
        return temp_min;
    }
    
    /* Does ray intersect planes in any direction at all? */
    bool alpha_max_exists( coordRay ray, coordGrid grid )
    {
        return    intersects(0,ray,grid)
               || intersects(1,ray,grid)
               || intersects(2,ray,grid);
    }

//   coord_type get_alpha_max( coordVector const alpha_dimmax )
//   {
//       return min( min( alpha_dimmax[0],
//                        alpha_dimmax[1] ),
//                   alpha_dimmax[2] );
//   }

    /* Get parameter of exit point, i.e. get maximum parameter of an
     * intersection of the ray with a plane that is adjacent to a voxel which
     * has a finite intersection length with that ray */
    // Correct results only for `alpha_max_exists(...) == true`!!!
    coord_type get_alpha_max( coordRay ray, coordGrid grid )
    {
        coord_type temp;
        coord_type temp_max;
        bool       not_first = false;
        for(int dim=0;dim<3;dim++)
        {
            if(intersects(dim,ray,grid))
            {
                temp = get_alpha_dimmax(dim,ray,grid);
                if(not_first)
                {
                    temp_max = std::min(temp_max,temp); // !
                } else
                {
                    temp_max = temp;
                    not_first = true;
                }
            }
        }
        return temp_max;
    }
    
    /* Get minimum of these two plane indices in direction dim:
     *  - of first intersected plane after the ray entered voxel space
     *  - of last intersected plane including the outer plane of voxel space */
    int get_i_dimmin( int dim, coordRay ray, coordGrid grid )
    {
        if(ray.get_start()[dim] < ray.get_end()[dim])
        {
            coord_type alpha_min =    get_alpha_min(ray,grid);
            coord_type alpha_dimmin = get_alpha_dimmin(dim,ray,grid);
            if(alpha_dimmin != alpha_min)
            {
                return ceil(phi_from_alpha(alpha_min,dim,ray,grid));
            } else
            {
                return 1;
            }
        } else
        {
            coord_type alpha_max    = get_alpha_max(ray,grid);
            coord_type alpha_dimmax = get_alpha_dimmax(dim,ray,grid);
            if(alpha_dimmax != alpha_max)
            {
                return ceil(phi_from_alpha(alpha_max,dim,ray,grid));
            } else
            {
                return 0;
            }
        }
    }
    
    /* Get maximum of these two plane indices in direction dim:
     *  - of first intersected plane after the ray entered voxel space
     *  - of last intersected plane including the outer plane of voxel space */
    int get_i_dimmax( int dim, coordRay ray, coordGrid grid )
    {
        if(ray.get_start()[dim] < ray.get_end()[dim])
        {
            coord_type alpha_max    = get_alpha_max(ray,grid);
            coord_type alpha_dimmax = get_alpha_dimmax(dim,ray,grid);
            if(alpha_dimmax != alpha_max)
            {
                return floor(phi_from_alpha(alpha_max,dim,ray,grid));
            } else
            {
                return grid.get_N()[dim];
            }
        } else
        {
            coord_type alpha_min    = get_alpha_min(ray,grid);
            coord_type alpha_dimmin = get_alpha_dimmin(dim,ray,grid);
            if(alpha_dimmin != alpha_min)
            {
                return floor(phi_from_alpha(alpha_min,dim,ray,grid));
            } else
            {
                return grid.get_N()[dim]-1;
            }
        }
    }
    
    /* Get total number of grid planes crossed by the ray.  Actual number of
     * intersections might be smaller.  This is the case, if one or more
     * intersections cross more than one plane at the same point. */    
    int get_N_crossed_planes( coordRay ray, coordGrid grid )
    {
      int i_max[3] = {get_i_dimmax(0, ray, grid),
                      get_i_dimmax(1, ray, grid),
                      get_i_dimmax(2, ray, grid)
                     };
      int i_min[3] = {get_i_dimmin(0, ray, grid),
                      get_i_dimmin(1, ray, grid),
                      get_i_dimmin(2, ray, grid)
                     };
      return   i_max[0] - i_min[0]\
             + i_max[1] - i_min[1]\
             + i_max[2] - i_min[2]\
             + 3;
    }
    
    /* Helper function for implementation of improved Siddon algorithm by Jacobs
     * et alii. */
    void update_alpha_dim( coord_type & alpha_dim, int dim, coordRay ray, coordGrid grid )
    {
      alpha_dim += grid.get_difference()[dim]/std::abs(ray.get_end()[dim]-ray.get_start()[dim]);
    }

    /* Helper function for implementation of improved Siddon algorithm by Jacobs
     * et alii. */
    void update_i_dim( int & i_dim, int dim, coordRay ray, coordGrid grid )
    {
      int i_update;
      if(ray.get_start()[dim]<ray.get_end()[dim]){
        i_update = 1;
      } else {
        i_update = -1;
      }
      i_dim += i_update;
    }
    
    /* Implementation of the improved Siddon algorithm by Jacobs et alii. */
    void calculate_intersection_lengths( coord_type * a, coordRay ray, coordGrid grid )
    {
      coord_type length =     ray.get_length();
      coord_type alpha_min =  get_alpha_min(ray, grid);
      coord_type alpha_max =  get_alpha_max(ray, grid);
      int i_x_min =           get_i_dimmin(0, ray, grid);
      int i_x_max =           get_i_dimmax(0, ray, grid);
      int i_y_min =           get_i_dimmin(1, ray, grid);
      int i_y_max =           get_i_dimmax(1, ray, grid);
      int i_z_min =           get_i_dimmin(2, ray, grid);
      int i_z_max =           get_i_dimmax(2, ray, grid);
      
      coord_type alpha_x;
      if(ray.get_end()[0] > ray.get_start()[0]) {
        alpha_x =             alpha_from_i(i_x_min, 0, ray, grid);
      } else{
        alpha_x =             alpha_from_i(i_x_max, 0, ray, grid);
      }
      coord_type alpha_y;
      if(ray.get_end()[1] > ray.get_start()[1]) {
        alpha_y =             alpha_from_i(i_y_min, 1, ray, grid);
      } else {
        alpha_y =             alpha_from_i(i_y_max, 1, ray, grid);
      }
      coord_type alpha_z;
      if(ray.get_end()[2] > ray.get_start()[2]) {
        alpha_z =             alpha_from_i(i_z_min, 2, ray, grid);
      } else {
        alpha_z =             alpha_from_i(i_z_max, 2, ray, grid);
      }
      
      int i_x =               std::floor(phi_from_alpha(((min(alpha_x, alpha_y, alpha_z)+alpha_min)/2.), 0, ray, grid));
      int i_y =               std::floor(phi_from_alpha(((min(alpha_x, alpha_y, alpha_z)+alpha_min)/2.), 1, ray, grid));
      int i_z =               std::floor(phi_from_alpha(((min(alpha_x, alpha_y, alpha_z)+alpha_min)/2.), 2, ray, grid));
      
      coord_type alpha_curr = alpha_min;

#ifdef DEBUG
      std::cout << "length:      " << length     << std::endl;
      std::cout << "alpha_min :  " << alpha_min  << std::endl;
      std::cout << "alpha_max :  " << alpha_max  << std::endl;
      std::cout << std::endl;
      std::cout << "i_x_min :    " << i_x_min    << std::endl;
      std::cout << "i_y_min :    " << i_y_min    << std::endl;
      std::cout << "i_z_min :    " << i_z_min    << std::endl;
      std::cout << std::endl;
      std::cout << "alpha_x :    " << alpha_x    << std::endl;
      std::cout << "alpha_y :    " << alpha_y    << std::endl;
      std::cout << "alpha_z :    " << alpha_z    << std::endl;
      std::cout << "alpha_curr : " << alpha_curr << std::endl;
      std::cout << std::endl;
#endif

      // Iterate
      int i = 0;
      while(alpha_curr < alpha_max){
        if(     alpha_x == min(alpha_x, alpha_y, alpha_z))
        {
#ifdef DEBUG
          std::cout << "intersect x plane at alpha = " << alpha_x << std::endl;
#endif
          a[i] = (alpha_x - alpha_curr)*length;
          update_i_dim(i_x, 0, ray, grid);
          alpha_curr = alpha_x;
          update_alpha_dim(alpha_x, 0, ray, grid);
          
          if(alpha_curr == alpha_y)
          {
            update_i_dim(i_y, 1, ray, grid);
            update_alpha_dim(alpha_y, 1, ray, grid);
          }
          if(alpha_curr == alpha_z)
          {
            update_i_dim(i_z, 2, ray, grid);
            update_alpha_dim(alpha_z, 2, ray, grid);
          }
        }
        else if(alpha_y == min(alpha_x, alpha_y, alpha_z))
        {
#ifdef DEBUG
          std::cout << "intersect y plane at alpha = " << alpha_y << std::endl;
#endif
          a[i] = (alpha_y - alpha_curr)*length;
          update_i_dim(i_y, 1, ray, grid);
          alpha_curr = alpha_y;
          update_alpha_dim(alpha_y, 1, ray, grid);
          
          if(alpha_curr == alpha_z)
          {
            update_i_dim(i_z, 2, ray, grid);
            update_alpha_dim(alpha_z, 2, ray, grid);
          }
        }
        else
        {
#ifdef DEBUG
          std::cout << "intersect z plane at alpha = " << alpha_z << std::endl;
#endif
          a[i] = (alpha_z - alpha_curr)*length;
          update_i_dim(i_z, 2, ray, grid);
          alpha_curr = alpha_z;
          update_alpha_dim(alpha_z, 2, ray, grid);
        }
        i++;
      }
    }
}

#endif  // #ifndef SIDDON_HPP
