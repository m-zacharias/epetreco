#include "Siddon.hpp"
#include "Vertex.hpp"
#include <iostream>
#include <cmath>


class coordRay : public Ray<coord_type>
{
  public:
    
    coordRay( Vertex const start, Vertex const end )
    : _start(start), _end(end) {}
    
    virtual Vertex start() const
    {
      return _start;
    }

    virtual Vertex end() const
    {
      return _end;
    }

    virtual coord_type length() const
    {
      return std::sqrt(
                        (_end.x-_start.x)*(_end.x-_start.x) +
                        (_end.y-_start.y)*(_end.y-_start.y) +
                        (_end.z-_start.z)*(_end.z-_start.z)
             );
    }


  protected:
    
    Vertex _start, _end;
};

class coordGrid : public Grid<coord_type>
{
  public:
    
    coordGrid( Vertex const origin,
               Vertex const diff,
               int Nx, int Ny, int Nz )
    : _origin(origin), _diff(diff), _Nx(Nx), _Ny(Ny), _Nz(Nz) {}

    Vertex origin() const
    {
      return _origin;
    }

    Vertex diff() const
    {
      return _diff;
    }

    int Nx() const
    {
      return _Nx;
    }

    int Ny() const
    {
      return _Ny;
    }

    int Nz() const
    {
      return _Nz;
    }


  protected:
    
    Vertex _origin, _diff;

    int _Nx, _Ny, _Nz;
};


using namespace Siddon;

int main( void ) {
    coordRay  ray(  Vertex(2.5,2.5,2.5),
                    Vertex(1.5,1.5,1.5)
              );

    coordGrid grid( Vertex(1.,1.,1.),
                    Vertex(1.,1.,1.),
                    3,3,3
              );

    std::cout << "alpha_xmin: " << get_alpha_dimmin__x< coordRay,coordGrid> (ray,grid) << std::endl
              << "alpha_min: "  << get_alpha_min<       coordRay,coordGrid> (ray,grid) << std::endl
              << "i_xmin: "     << get_i_dimmin__x<     coordRay,coordGrid> (ray,grid) << std::endl;
    std::cout << "alpha_xmax: " << get_alpha_dimmax__x< coordRay,coordGrid> (ray,grid) << std::endl
              << "alpha_max: "  << get_alpha_max<       coordRay,coordGrid> (ray,grid) << std::endl
              << "i_xmax: "     << get_i_dimmax__x<     coordRay,coordGrid> (ray,grid) << std::endl;

    return 0;
}
