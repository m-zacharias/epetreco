#ifndef DEVELRAY_HPP
#define DEVELRAY_HPP

#include <cmath>
#include "Ply.hpp"
#include "Ray.hpp"
#include "TemplateVertex.hpp"

#ifdef DEBUG
#include <iostream>
#endif

typedef double CoordType;


struct DevelRayTraits
{
  typedef TemplateVertex<CoordType> Vertex_t;
};


class DevelRay : public Ray<DevelRay, DevelRayTraits>,
                 public PlyLine<typename DevelRayTraits::Vertex_t>
{
  public:
    
    // Constructor
    DevelRay( Vertex_t const p0, Vertex_t const p1 )
    : PlyLine<Vertex_t>(std::string(""),p0,p1)
    {
#ifdef DEBUG
#ifndef NO_DEVELRAY_DEBUG
      std::cout << "DevelRay::DevelRay(Vertex_t const,Vertex_t const)"
                << std::endl;
#endif
#endif
    }
    
    // Default constructor
    DevelRay( void )
    : PlyLine<Vertex_t>(std::string(""),Vertex_t(0,0,0),Vertex_t(0,0,0))
    {
#ifdef DEBUG
#ifndef NO_DEVELRAY_DEBUG
      std::cout << "DevelRay::DevelRay()"
                << std::endl;
#endif
#endif
    }
        
    Vertex_t start() const
    {
#ifdef DEBUG
#ifndef NO_DEVELRAY_DEBUG
      std::cout << "DevelRay::start()"
                << std::endl;
#endif
#endif
      return PlyLine<Vertex_t>::_p0;
    }
    
    Vertex_t end() const
    {
#ifdef DEBUG
#ifndef NO_DEVELRAY_DEBUG
      std::cout << "DevelRay::end()"
                << std::endl;
#endif
#endif
      return PlyLine<Vertex_t>::_p1;
    }
    
    Vertex_t::Coord_t length() const
    {
#ifdef DEBUG
#ifndef NO_DEVELRAY_DEBUG
      std::cout << "DevelRay::"
                << std::endl;
#endif
#endif
      return std::sqrt( (_p1.x-_p0.x)*(_p1.x-_p0.x) +
                        (_p1.y-_p0.y)*(_p1.y-_p0.y) +
                        (_p1.z-_p0.z)*(_p1.z-_p0.z)
             );
    }
};

#endif  // #ifndef DEVELRAY_HPP
