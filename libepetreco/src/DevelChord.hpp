/** @file DevelChord.hpp */
#ifndef DEVELCHORD_HPP
#define DEVELCHORD_HPP

#include "Chord.hpp"
#include "DevelGrid.hpp"
#include "DevelRay.hpp"

struct DevelChordTraits
{
  typedef double    Coord_t;
  typedef DevelGrid  Grid_t;
  typedef DevelRay   Ray_t;
};


class DevelChord : public Chord<DevelChord, DevelChordTraits>
{
  private:
    
    int _id[3];

    Coord_t _length;


  public:
    
    void setId( int const * const id )
    {
#if ((defined DEBUG || defined DEBUG_DEVELCHORD) && (NO_DEVELCHORD_DEBUG==0))
      std::cout << "DevelChord::seId(int const* const)"
                << std::endl;
#endif
      for(int dim=0; dim<3; dim++)
        _id[dim] = id[dim];
    }

    void setLength( Coord_t const & length )
    {
#if ((defined DEBUG || defined DEBUG_DEVELCHORD) && (NO_DEVELCHORD_DEBUG==0))
      std::cout << "DevelChord::setLength(Coord_t const&)"
                << std::endl;
#endif
      _length = length;
    }

    void getId( int * const id ) const
    {
#if ((defined DEBUG || defined DEBUG_DEVELCHORD) && (NO_DEVELCHORD_DEBUG==0))
      std::cout << "DevelChord::getId(int* const)"
                << std::endl;
#endif
      for(int dim=0; dim<3; dim++)
        id[dim] = _id[dim];
    }

    Coord_t getLength( void ) const
    {
#if ((defined DEBUG || defined DEBUG_DEVELCHORD) && (NO_DEVELCHORD_DEBUG==0))
      std::cout << "DevelChord::getLength()"
                << std::endl;
#endif
      return _length;
    }

    int getLinearIdVoxel( Grid_t const & grid ) const
    {
#if ((defined DEBUG || defined DEBUG_DEVELCHORD) && (NO_DEVELCHORD_DEBUG==0))
      std::cout << "DevelChord::getLinearIdVoxel(Grid_t const&)"
                << std::endl;
#endif
      return  _id[2]\
            + _id[1] * grid.N()[2]\
            + _id[0] * grid.N()[2] * grid.N()[1];
    }
};

#endif  // #ifndef DEVELCHORD_HPP
