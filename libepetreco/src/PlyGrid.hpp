#ifndef PLYGRID_HPP
#define PLYGRID_HPP

#include "CompositePlyGeometry.hpp"

class PlyGrid : public CompositePlyGeometry
{
  public:
    
    PlyGrid( std::string const,
             Vertex,
             int const &, int const &, int const &,
             coord_type const &, coord_type const &, coord_type const & );
};

#endif  // #define PLYGRID_HPP
