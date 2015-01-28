/** @file Chord.hpp */
#ifndef CHORD_HPP
#define CHORD_HPP

template<typename ConcreteChord, typename ConcreteChordTraits>
class Chord
{
  public:
    
    typedef typename ConcreteChordTraits::Coord_t Coord_t;
    typedef typename ConcreteChordTraits::Grid_t  Grid_t;
    typedef typename ConcreteChordTraits::Ray_t   Ray_t;

    void setId( int const * const id )
    {
      static_cast<ConcreteChord *>(this)->setId(id);
    }

    void setLength( Coord_t const & length )
    {
      static_cast<ConcreteChord *>(this)->setLength(length);
    }

    void getId( int * const id ) const
    {
      static_cast<ConcreteChord *>(this)->getId(id);
    }

    Coord_t getLength( void ) const
    {
      return static_cast<ConcreteChord *>(this)->getLength();
    }

    int getLinearIdVoxel( Grid_t const & grid ) const
    {
      return static_cast<ConcreteChord *>(this)->getLinearIdVoxel(grid);
    }
};

#endif  // #ifndef CHORD_HPP
