#ifndef DEVELSETUP_HPP
#define DEVELSETUP_HPP

#include "DevelChannel.hpp"

#ifdef DEBUG
#include <iostream>
#endif

class DevelSetup
{
  public:
    
    typedef DevelChannel                Channel_t;
    typedef typename Channel_t::Coord_t Coord_t;
    
    
  private:
    
    int _N;


  public:
    
    DevelSetup( int const & N )
    : _N(N)
    {
#ifdef DEBUG
#ifndef NO_DEVELSETUP_DEBUG
      std::cout << "DevelSetup::DevelSetup(int const&)"
                << std::endl;
#endif
#endif
    }

    int getNChannels() const
    {
#ifdef DEBUG
#ifndef NO_DEVELSETUP_DEBUG
      std::cout << "DevelSetup::getNChannels()"
                << std::endl;
#endif
#endif
      return _N;
    }

    int getLinearChannelId(
          int angle, int det0segz, int det0segx, int det1segz,
          int det1segx ) const
    {
#ifdef DEBUG
#ifndef NO_DEVELSETUP_DEBUG
      std::cout << "DevelSetup::getLinearChannelId(int,int,int,int,int)"
                << std::endl;
#endif
#endif
      return   angle    *13*13*13*13\
             + det0segz *13*13*13\
             + det0segx *13*13\
             + det1segz *13
             + det1segx;
    }

    void getDimensionalChannelId(
          int const linearChannelId, int * const dimensionalChannelId ) const
    {
#ifdef DEBUG
#ifndef NO_DEVELSETUP_DEBUG
      std::cout << "DevelSetup::getDimensionalChannelId(int const,int* const)"
                << std::endl;
#endif
#endif
      dimensionalChannelId[0] =  linearChannelId    /(13*13*13*13);
      dimensionalChannelId[1] = (linearChannelId% 180) /(13*13*13);
      dimensionalChannelId[2] = (linearChannelId%(180*13))/(13*13);
      dimensionalChannelId[3] = (linearChannelId%(180*13*13)) /13;
      dimensionalChannelId[4] =  linearChannelId%(180*13*13*13);
    }

    void getPos0(
          int const * const dimensionalChannelId, Coord_t * const pos0 ) const
    {
#ifdef DEBUG
#ifndef NO_DEVELSETUP_DEBUG
      std::cout << "DevelSetup::getPos0(int const* const,Coord_t* const)"
                << std::endl;
#endif
#endif
      pos0[0] = -45.7;
      pos0[1] = (6-dimensionalChannelId[1])*.4;
      pos0[2] = (6-dimensionalChannelId[2])*.4;
    }

    void getPos1( int const * const dimensionalChannelId, Coord_t * const pos1 ) const
    {
#ifdef DEBUG
#ifndef NO_DEVELSETUP_DEBUG
      std::cout << "DevelSetup::int const* const,Coord_t* const"
                << std::endl;
#endif
#endif
      pos1[0] = 45.7;
      pos1[1] = (6-dimensionalChannelId[1])*.4;
      pos1[2] = (6-dimensionalChannelId[2])*.4;
    }

    void createChannels( Channel_t * const channels ) const
    {
#ifdef DEBUG
#ifndef NO_DEVELSETUP_DEBUG
      std::cout << "DevelSetup::createChannels(Channel_t* const)"
                << std::endl;
#endif
#endif
      int NChannels(getNChannels());
      for(int linearChannelId=0; linearChannelId<NChannels; linearChannelId++)
      {
        Coord_t pos0_[3];
        Coord_t pos1_[3];
        int dimensionalChannelId[5];
        getDimensionalChannelId(linearChannelId, dimensionalChannelId);
        getPos0(dimensionalChannelId, pos0_);
        getPos1(dimensionalChannelId, pos1_);
        
        channels[linearChannelId] = Channel_t(2*dimensionalChannelId[0], pos0_, pos1_);
      }
    }

    void createChannels(
          Channel_t * const channels, int const beginChannelId,
          int const endChannelId) const
    {
#if ((defined DEBUG || defined DEVELSETUP_DEBUG) && (NO_DEVELSETUP_DEBUG==0))
      std::cout << "DevelSetup::createChannels(Channel_t* const, int const, "
                << "int const)"
                << std::endl;
#endif
      //int NChannels(endChannelId-beginChannelId);
      for(int linearChannelId = beginChannelId;
              linearChannelId < endChannelId;
              linearChannelId++)
      {
        Coord_t pos0_[3];
        Coord_t pos1_[3];
        int dimensionalChannelId[5];
        getDimensionalChannelId(linearChannelId, dimensionalChannelId);
        getPos0(dimensionalChannelId, pos0_);
        getPos1(dimensionalChannelId, pos1_);
        
        channels[linearChannelId] = Channel_t(2*dimensionalChannelId[0], pos0_, pos1_);
      }
    }
};

#endif  // #ifndef DEVELSETUP_HPP
