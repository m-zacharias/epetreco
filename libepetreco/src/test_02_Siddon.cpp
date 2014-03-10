#include "Siddon.hpp"
#include "Ray.hpp"
#include "Grid.hpp"
#include "TemplateVertex.hpp"
#include "PlyGrid.hpp"
#include "PlyLine.hpp"
#include "PlyWriter.hpp"
#include <iostream>
#include <string>
#include <cmath>

typedef double CoordType;


struct CRayTraits
{
  typedef TemplateVertex<CoordType> Vertex_t;
};


class CRay : public Ray<CRay, CRayTraits>, public PlyLine<CRayTraits::Vertex_t>
{
  public:
    
    CRay( std::string const name,
          Vertex_t const p0, Vertex_t const p1 )
    : PlyLine<Vertex_t>(name,p0,p1) {}
    
    virtual Vertex_t start() const { return PlyLine<Vertex_t>::_p0; }
    
    virtual Vertex_t end() const { return PlyLine<Vertex_t>::_p1; }
    
    virtual Vertex_t::Coord_t length() const
    {
      return std::sqrt( (_p1.x-_p0.x)*(_p1.x-_p0.x) +
                        (_p1.y-_p0.y)*(_p1.y-_p0.y) +
                        (_p1.z-_p0.z)*(_p1.z-_p0.z)
             );
    }
};


struct CGridTraits
{
  typedef TemplateVertex<CoordType> Vertex_t;
};


class CGrid : public Grid<CGrid, CGridTraits>, public PlyGrid<CGridTraits::Vertex_t>
{
  public:
    
    CGrid( std::string const name,
           Vertex_t const origin,
           Vertex_t const diff,
           int const Nx, int const Ny, int const Nz )
    : PlyGrid<Vertex_t>(name,origin,Nx+1,Ny+1,Nz+1,diff.x,diff.y,diff.z),
      _origin(origin), _diff(diff),
      _Nx(Nx), _Ny(Ny), _Nz(Nz) {}
      
    virtual Vertex_t origin() const { return _origin; }

    virtual Vertex_t diff( void ) const { return _diff; }

    virtual int Nx() const { return _Nx; }

    virtual int Ny() const { return _Ny; }

    virtual int Nz() const { return _Nz; }


  private:
    
    Vertex_t _origin, _diff;

    int _Nx, _Ny, _Nz;
};



using namespace Siddon;

int main( void )
{
    // Make Objects
    CRay ray1(  "ray1",
                CRay::Vertex_t(-0.5, 0.1, 0.1),
                CRay::Vertex_t( 5.5, 0.9, 0.9)
         );

    CGrid grid( "grid",
                CRay::Vertex_t(0.,0.,0.),
                CRay::Vertex_t(1.,1.,1.),
                5,1,1
          );
    
    // Prepare intersection array
    int N_crossed = get_N_crossed_planes<CRay,CGrid>(ray1, grid);
    Intersection<CoordType> a[N_crossed];

    for(int i=0; i<N_crossed; i++) {
      a[i].length = 0;
      a[i].idx = -1;
      a[i].idy = -1;
      a[i].idz = -1;
    }

    // Intersect
    calculate_intersection_lengths<CRay,CGrid>(a, ray1, grid);

    for(int i=0; i<N_crossed; i++) {
      std::cout << a[i].idx << " " << a[i].idy << " " << a[i].idz << " : " << a[i].length << std::endl;
    }
    std::cout << std::endl;
    
    // Write Visualisation
    PlyWriter grid_writer("test2_grid_output.ply");
    std::cout << "grid_writer.write() ..." << std::endl;
    grid_writer.write(grid);
    std::cout << "grid_writer.close() ..." << std::endl;
    grid_writer.close();
    std::cout << "grid_writer success" << std::endl;

    PlyWriter ray1_writer("test2_ray1_output.ply");
    ray1_writer.write(ray1);
    ray1_writer.close();
    std::cout << "PlyWriter success" << std::endl;

    return 0;
}
