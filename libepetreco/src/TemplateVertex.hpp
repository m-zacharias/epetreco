#ifndef TEMPLATEVERTEX_HPP
#define TEMPLATEVERTEX_HPP

template<typename T>
struct TemplateVertex {

  public:

    typedef T Coord_t;
    
    /* Parametric Constructor */
    TemplateVertex( T const x_, T const y_, T const z_ );
    
    /* Copy Constructor */
    TemplateVertex( TemplateVertex const & v );

    /* Copy Assignment */
    void operator=( TemplateVertex const & v );
    


    T x, y, z;
};
#include "TemplateVertex.tpp"

// typedef double coord_type;
// typedef TemplateVertex<coord_type> Vertex;

#endif  // #define TEMPLATEVERTEX_HPP
