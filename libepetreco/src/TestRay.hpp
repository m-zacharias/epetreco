#ifndef TESTRAY_HPP
#define TESTRAY_HPP

#include "Ply.hpp"
#include "Ray.hpp"
#include "TemplateVertex.hpp"

typedef double CoordType;


struct TestRayTraits
{
  typedef TemplateVertex<CoordType> Vertex_t;
};


class TestRay : public Ray<TestRay, TestRayTraits>,
                public PlyLine<typename TestRayTraits::Vertex_t>
{
  public:
    
    // Constructor
    TestRay( Vertex_t const p0, Vertex_t const p1 )
    : PlyLine<Vertex_t>(std::string(""),p0,p1) {}
    
    // Default constructor
    TestRay( void )
    : PlyLine<Vertex_t>(std::string(""),Vertex_t(0,0,0),Vertex_t(0,0,0)) {}
        
    Vertex_t start() const { return PlyLine<Vertex_t>::_p0; }
    
    Vertex_t end() const { return PlyLine<Vertex_t>::_p1; }
    
    Vertex_t::Coord_t length() const
    {
      return std::sqrt( (_p1.x-_p0.x)*(_p1.x-_p0.x) +
                        (_p1.y-_p0.y)*(_p1.y-_p0.y) +
                        (_p1.z-_p0.z)*(_p1.z-_p0.z)
             );
    }
};

#endif  // #ifndef TESTRAY_HPP
