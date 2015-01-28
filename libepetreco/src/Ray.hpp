/** @file Ray.hpp */
#ifndef RAY_HPP
#define RAY_HPP

/* Abstract Ray class */
template<typename ConcreteRay, typename ConcreteRayTraits>
struct Ray
{  
  public:
    
    typedef typename ConcreteRayTraits::Vertex_t Vertex_t;

    Vertex_t start() const
    {
      return static_cast<ConcreteRay *>(this)->start();
    }
    
    Vertex_t end() const
    {
      return static_cast<ConcreteRay *>(this)->end();
    }

    typename Vertex_t::Coord_t length()
    {
      return static_cast<ConcreteRay *>(this)->length();
    }
};

#endif  // #ifndef RAY_HPP
