#ifndef TESTGRID_HPP
#define TESTGRID_HPP

#include "Ply.hpp"
#include "Grid.hpp"
#include "TemplateVertex.hpp"

typedef double CoordType;


struct TestGridTraits
{
  typedef TemplateVertex<CoordType> Vertex_t;
};


class TestGrid : public Grid<TestGrid, TestGridTraits>,
                 public PlyGrid<TestGridTraits::Vertex_t>
{
  public:
    
    TestGrid(
           Vertex_t const origin,
           Vertex_t const diff,
           int const Nx, int const Ny, int const Nz )
    : PlyGrid<Vertex_t>(std::string(""),origin,Nx+1,Ny+1,Nz+1,diff.x,diff.y,diff.z),
      _origin(origin), _diff(diff),
      _Nx(Nx), _Ny(Ny), _Nz(Nz) {}
      
    Vertex_t origin() const { return _origin; }

    Vertex_t diff( void ) const { return _diff; }

    TemplateVertex<int> N( void ) const { return TemplateVertex<int>(Nx(),Ny(),Nz()); }

    int Nx() const { return _Nx; }

    int Ny() const { return _Ny; }

    int Nz() const { return _Nz; }


  private:
    
    Vertex_t _origin, _diff;

    int _Nx, _Ny, _Nz;
};

#endif  // #ifndef TESTGRID_HPP
