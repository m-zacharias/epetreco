#include "Siddon.hpp"
#include "Vertex.hpp" // typedef coord_type
#include "PlyGrid.hpp"
#include "PlyLine.hpp"
#include "PlyWriter.hpp"
#include <iostream>
#include <string>
#include <cmath>

class CRay : public Ray<coord_type>, public PlyLine
{
  public:
    
    CRay( std::string const name,
            Vertex const p0, Vertex const p1 )
    : PlyLine(name,p0,p1) {}
    
    virtual Vertex start() const
    {
      return PlyLine::_p0;
    }
    
    virtual Vertex end() const
    {
      return PlyLine::_p1;
    }
    
    virtual coord_type length() const
    {
      return std::sqrt( (_p1.x-_p0.x)*(_p1.x-_p0.x) +
                        (_p1.y-_p0.y)*(_p1.y-_p0.y) +
                        (_p1.z-_p0.z)*(_p1.z-_p0.z)
             );
    }
};



class CGrid : public Grid<coord_type>, public PlyGrid
{
  public:
    
    CGrid( std::string const name,
           Vertex const origin,
           TemplateVertex<coord_type> const diff,
           int const Nx, int const Ny, int const Nz )
    : PlyGrid(name,origin,Nx,Ny,Nz,diff.x,diff.y,diff.z),
      _origin(origin), _diff(diff),
      _Nx(Nx), _Ny(Ny), _Nz(Nz) {}
      
    virtual Vertex origin() const
    {
      return _origin;
    }

    virtual Vertex diff( void ) const
    {
      return _diff;
    }

    virtual int Nx() const
    {
      return _Nx;
    }

    virtual int Ny() const
    {
      return _Ny;
    }

    virtual int Nz() const
    {
      return _Nz;
    }


  private:
    
    Vertex _origin, _diff;

    int _Nx, _Ny, _Nz;
};



using namespace Siddon;

int main( void )
{
    // Make Objects
    CRay ray1(  "ray1",
                Vertex(-0.5, 0.5, 0.5),
                Vertex( 1.5, 0.5, 0.5)
         );

    CGrid grid( "grid",
                Vertex(0.,0.,0.),
                Vertex(1.,1.,1.),
                2,2,2
          );
    
    // Prepare intersection array
    int N_crossed = get_N_crossed_planes<CRay,CGrid>(ray1, grid);
    Intersection<coord_type> a[N_crossed];

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
    grid_writer.write(grid);
    grid_writer.close();

    PlyWriter ray1_writer("test2_ray1_output.ply");
    ray1_writer.write(ray1);
    ray1_writer.close();

    return 0;
}
