#ifndef VERTEX_HPP
#define VERTEX_HPP

template<typename T>
struct TemplateVertex {

  public:
    
    /* Parametric Constructor */
    TemplateVertex( T const x_, T const y_, T const z_ );
    
    /* Copy Constructor */
    TemplateVertex( TemplateVertex const & v );

    /* Copy Assignment */
    void operator=( TemplateVertex const & v );
    


    T x, y, z;
};
#include "Vertex.tpp"

typedef double coord_type;
typedef TemplateVertex<coord_type> Vertex;

#endif  // #define VERTEX_HPP
