#ifndef DEVELSMCHUNK_HPP
#define DEVELSMCHUNK_HPP

#include "DevelSetup.hpp"
#include "DevelGrid.hpp"

#if ((defined DEBUG || defined DEVELSMCHUNK_DEBUG) && (NO_DEVELSMCHUNK_DEBUG==0))
#include <iostream>
#endif

class DevelSMChunk
{
  public:
    
    typedef double Elem_t;
    
    
  private:
    
    Elem_t * _mtx;
    int _N, _M;
    
    
  public:
    
    DevelSMChunk( DevelSetup const & setup, DevelGrid const & grid )
    : _N(setup.getNChannels()), _M(grid.getNVoxels())
    {
#if ((defined DEBUG || defined DEVELSMCHUNK_DEBUG) && (NO_DEVELSMCHUNK_DEBUG==0))
      std::cout << "DevelSMChunk::DevelChunk(DevelSetup const&, "
                << "DevelGrid const&)"
                << std::endl;
#endif
      _mtx = new Elem_t[_N*_M];
    }

    ~DevelSMChunk()
    {
#if ((defined DEBUG || defined DEVELSMCHUNK_DEBUG) && (NO_DEVELSMCHUNK_DEBUG==0))
      std::cout << "DevelSMChunk::~DevelSMChunk()"
#endif
      delete[] _mtx;
    }

    void setElem( int linearChannelId, int linearVoxelId, Elem_t const val )
    {
#if ((defined DEBUG || defined DEVELSMCHUNK_DEBUG) && (NO_DEVELSMCHUNK_DEBUG==0))
      std::cout << "DevelSMChunk::setElem(int,int,Elem_t const)"
                << std::endl;
#endif
      _mtx[linearChannelId*_M + linearVoxelId] = val;
    }

    Elem_t getElem( int linearChannelId, int linearVoxelId ) const
    {
#if ((defined DEBUG || defined DEVELSMCHUNK_DEBUG) && (NO_DEVELSMCHUNK_DEBUG==0))
      std::cout << "DevelSMChunk::getElem(int,int)"
                << std::endl;
#endif
      return _mtx[linearChannelId*_M + linearVoxelId];
    }

    int getNVoxel( void ) const
    {
#if ((defined DEBUG || defined DEVELSMCHUNK_DEBUG) && (NO_DEVELSMCHUNK_DEBUG==0))
      std::cout << "DevelSMChunk::getNVoxel()"
                << std::endl;
#endif
      return _M;
    }

    int getNChannel( void ) const
    {
#if ((defined DEBUG || defined DEVELSMCHUNK_DEBUG) && (NO_DEVELSMCHUNK_DEBUG==0))
      std::cout << "DevelSMChunk::getNChannel()"
                << std::endl;
#endif
      return _N;
    }
};

#endif  // #ifndef DEVELSMCHUNK_HPP
