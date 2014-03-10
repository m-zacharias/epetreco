#include "Siddon.hpp"
#include "Ray.hpp"
#include "Grid.hpp"
#include "TemplateVertex.hpp"
#include <iostream>
#include <cmath>

typedef double CoordType;


struct CoordRayTraits
{
  typedef TemplateVertex<CoordType> Vertex_t;
};


class CoordRay : public Ray<CoordRay, CoordRayTraits>
{
  public:
    
    CoordRay( Vertex_t const start, Vertex_t const end )
    : _start(start), _end(end) {}
    
    Vertex_t start() const { return _start; }

    Vertex_t end() const { return _end; }

    typename Vertex_t::Coord_t length() const
    {
      return std::sqrt(
                        (_end.x-_start.x)*(_end.x-_start.x) +
                        (_end.y-_start.y)*(_end.y-_start.y) +
                        (_end.z-_start.z)*(_end.z-_start.z)
             );
    }


  protected:
    
    Vertex_t _start, _end;
};


struct CoordGridTraits
{
  typedef TemplateVertex<CoordType> Vertex_t;
};


class CoordGrid : public Grid<CoordGrid, CoordGridTraits>
{
  public:
    
    CoordGrid( Vertex_t const origin,
               Vertex_t const diff,
               int Nx, int Ny, int Nz )
    : _origin(origin), _diff(diff), _Nx(Nx), _Ny(Ny), _Nz(Nz) {}

    Vertex_t origin() const { return _origin; }

    Vertex_t diff() const { return _diff; }

    int Nx() const { return _Nx; }

    int Ny() const { return _Ny; }

    int Nz() const { return _Nz; }


  protected:
    
    Vertex_t _origin, _diff;

    int _Nx, _Ny, _Nz;
};


using namespace Siddon;

int main( void ) {
    CoordRay  ray(  CoordRay::Vertex_t(2.5,2.5,2.5),
                    CoordRay::Vertex_t(1.5,1.5,1.5)
              );

    CoordGrid grid( CoordGrid::Vertex_t(1.,1.,1.),
                    CoordGrid::Vertex_t(1.,1.,1.),
                    3,3,3
              );

    std::cout << "alpha_xmin: " << get_alpha_dimmin__x< CoordRay,CoordGrid> (ray,grid) << std::endl
              << "alpha_min: "  << get_alpha_min<       CoordRay,CoordGrid> (ray,grid) << std::endl
              << "i_xmin: "     << get_i_dimmin__x<     CoordRay,CoordGrid> (ray,grid) << std::endl;
    std::cout << "alpha_xmax: " << get_alpha_dimmax__x< CoordRay,CoordGrid> (ray,grid) << std::endl
              << "alpha_max: "  << get_alpha_max<       CoordRay,CoordGrid> (ray,grid) << std::endl
              << "i_xmax: "     << get_i_dimmax__x<     CoordRay,CoordGrid> (ray,grid) << std::endl;

    return 0;
}
