#ifndef VERTEX_HPP
#define VERTEX_HPP

typedef double coord_type;

struct Vertex
{
  public:
    
    Vertex( coord_type, coord_type, coord_type );
    Vertex( Vertex const & );

    void operator=( Vertex const & );
    
    coord_type x, y, z;    
};

#endif  // #define VERTEX_HPP

