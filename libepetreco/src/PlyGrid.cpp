#include "PlyGrid.hpp"
#include "PlyRectangle.hpp"

#ifdef DEBUG
#include <iostream>
#endif

PlyGrid::PlyGrid( std::string const name,
                  Vertex o,
                  int const & nx, int const & ny, int const & nz,
                  coord_type const & dx, coord_type const & dy, coord_type const & dz )
: CompositePlyGeometry(name)
{
  for(int idx=0; idx<nx; idx++) {
    this->add(new PlyRectangle(std::string(""),
                               Vertex(o.x + idx*dx,  o.y,           o.z),
                               Vertex(o.x + idx*dx,  o.y+(ny-1)*dy, o.z),
                               Vertex(o.x + idx*dx,  o.y+(ny-1)*dy, o.z+(nz-1)*dz),
                               Vertex(o.x + idx*dx,  o.y,           o.z+(nz-1)*dz)
                              )
             );
  }
  for(int idy=0; idy<ny; idy++) {
    this->add(new PlyRectangle(std::string(""),
                               Vertex(o.x,           o.y + idy*dy,  o.z),
                               Vertex(o.x,           o.y + idy*dy,  o.z+(nz-1)*dz),
                               Vertex(o.x+(nx-1)*dx, o.y + idy*dy,  o.z+(nz-1)*dz),
                               Vertex(o.x+(nx-1)*dx, o.y + idy*dy,  o.z)
                              )
             );
  }
  for(int idz=0; idz<nz; idz++) {
    this->add(new PlyRectangle(std::string(""),
                               Vertex(o.x,           o.y,           o.z + idz*dz),
                               Vertex(o.x+(nx-1)*dx, o.y,           o.z + idz*dz),
                               Vertex(o.x+(nx-1)*dx, o.y+(ny-1)*dy, o.z + idz*dz),
                               Vertex(o.x,           o.y+(ny-1)*dy, o.z + idz*dz)
                              )
             );
  }
}
