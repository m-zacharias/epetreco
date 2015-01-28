/** @file test_03_Siddon.cpp */
/* This test file illuminates the behavior of the intersection length
 * calculation for setups where the start and end of a ray are positioned within
 * the grid.
 * 
 * The setup:
 *  - grid of 3 cubes, aligned in x direction.  Origin (0,0,0), edge length 1.0
 *  - ray: Start (0.5,0.5,0.5), end (2.5,0.5,0.5)
 * 
 * Result as of 2014-03-10 :
 * ##################################
 * #  idx idy idz intersection_length
 * #  0 0 0   1
 * #  1 0 0   1
 * #  2 0 0   1
 * ##################################
 * (Obviously, it should read '1 0 0   1' and zero for all other voxels.
 * TODO: Fix this)
 */
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
    // Make ray, grid
    CRay ray1(  "ray1",
                CRay::Vertex_t( 0.5, 0.5, 0.5),
                CRay::Vertex_t( 2.5, 0.5, 0.5)
         );

    CGrid grid( "grid",
                CRay::Vertex_t(0.,0.,0.),
                CRay::Vertex_t(1.,1.,1.),
                3,1,1
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

    std::cout << "test_03_Siddon : idx idy idz intersection_length"
              << std::endl;
    for(int i=0; i<N_crossed; i++) {
      std::cout << "test_03_Siddon : "
                << a[i].idx << " " << a[i].idy << " " << a[i].idz
                << "   " << a[i].length << std::endl;
    }
    std::cout << std::endl;
    

    // Write Visualisation
    PlyWriter grid_writer("grid_output.ply");
    grid_writer.write(grid);
    grid_writer.close();

    PlyWriter ray1_writer("ray1_output.ply");
    ray1_writer.write(ray1);
    ray1_writer.close();

    return 0;
}
