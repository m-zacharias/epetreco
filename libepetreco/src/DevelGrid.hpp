#ifndef DEVELGRID_HPP
#define DEVELGRID_HPP

#include "Ply.hpp"
#include "Grid.hpp"
#include "TemplateVertex.hpp"

typedef double CoordType;


struct DevelGridTraits
{
  typedef TemplateVertex<CoordType> Vertex_t;
};


class DevelGrid : public Grid<DevelGrid, DevelGridTraits>,
                  public PlyGrid<DevelGridTraits::Vertex_t>
{
  public:
    
    DevelGrid(
           Vertex_t const origin,
           Vertex_t const diff,
           int const Nx, int const Ny, int const Nz )
    : PlyGrid<Vertex_t>(std::string(""),origin,Nx+1,Ny+1,Nz+1,diff.x,diff.y,diff.z),
      _origin(origin), _diff(diff),
      _Nx(Nx), _Ny(Ny), _Nz(Nz) {}
      
    Vertex_t origin() const { return _origin; }

    Vertex_t diff( void ) const { return _diff; }

    TemplateVertex<int> N( void ) const { return TemplateVertex<int>(Nx(),Ny(),Nz()); }

    int getNVoxels() const { return _Nx*_Ny*_Nz; }

    int Nx() const { return _Nx; }

    int Ny() const { return _Ny; }

    int Nz() const { return _Nz; }


  private:
    
    Vertex_t _origin, _diff;

    int _Nx, _Ny, _Nz;
};

#endif  // #ifndef DEVELGRID_HPP
