/** @file Grid.hpp */
#ifndef GRID_HPP
#define GRID_HPP

/* Abstract Grid class */
template<typename ConcreteGrid, typename ConcreteGridTraits>
struct Grid
{
  public:
    
    typedef typename ConcreteGridTraits::Vertex_t Vertex_t;

    Vertex_t origin() const
    {
      return static_cast<ConcreteGrid *>(this)->origin();
    }
    
    Vertex_t diff() const
    {
      return static_cast<ConcreteGrid *>(this)->diff();
    }

    int Nx() const
    {
      return static_cast<ConcreteGrid *>(this)->Nx();
    }

    int Ny() const
    {
      return static_cast<ConcreteGrid *>(this)->Ny();
    }

    int Nz() const
    {
      return static_cast<ConcreteGrid *>(this)->Nz();
    }
};

#endif  // #ifndef GRID_HPP
