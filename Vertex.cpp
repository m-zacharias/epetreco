#include "Vertex.hpp"

#ifdef DEBUG
#include <iostream>
#endif

Vertex::Vertex( coord_type x_, coord_type y_, coord_type z_ )
: x(x_), y(y_), z(z_)
{
#ifdef DEBUG
  std::cout << "Vertex::Vertex(coord_type, coord_type, coord_type)" << std::endl;
#endif
}

Vertex::Vertex( Vertex const & v )
: x(v.x), y(v.y), z(v.z)
{
#ifdef DEBUG
  std::cout << "Vertex::Vertex(Vertex const &)" << std::endl;
#endif
}

void Vertex::operator=( Vertex const & v )
{
#ifdef DEBUG
  std::cout << "Vertex::operator=(Vertex const &)" << std::endl;
#endif
  x=v.x; y=v.y; z=v.z;
}
