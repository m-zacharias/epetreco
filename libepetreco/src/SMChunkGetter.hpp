#ifndef SMCHUNKGETTER_HPP
#define SMCHUNKGETTER_HPP

#include "ChordsCalc.hpp"
#include "DevelRay.hpp"
#include "DevelGrid.hpp"
#include "DevelChord.hpp"

#ifdef DEBUG
#include <iostream>
#endif

typedef ChordsCalc<DevelRay, DevelGrid, DevelChord> ChordsCalcType;

template<typename Setup, typename Grid, typename SMChunk>
struct SMChunkGetter
{
  typedef typename Setup::Channel_t               Channel_t;
  typedef typename Channel_t::Ray_t               Ray_t;
  typedef          ChordsCalcType                 ChordsCalc_t;
  typedef typename ChordsCalc_t::Chord_t          Chord_t;

  void operator()(
             Setup const & setup, Grid const & grid, int const id,
             SMChunk & sm, int const nRays )
  {
#ifdef DEBUG
    std::cout << "SMChunkGetter<>::operator()(...)" << std::endl;
#endif
    // Create channels
    int nChannels = setup.getNChannels();
    Channel_t * channels = new Channel_t[nChannels];
    setup.createChannels(channels);
    
    // Iterate over channels ...
    for(int idChannel=0; idChannel<nChannels; idChannel++)
    {
      // Create rays in channel
      Ray_t * rays = new Ray_t[nRays];
      channels[idChannel].createRays(rays, nRays);
      
      // Iterate over rays ...
      for(int idRay=0; idRay<nRays; idRay++)
      {
        // Calculate chords
        ChordsCalc_t calc;
        int nA = calc.getNChords(rays[idRay], grid);
        Chord_t * a = new Chord_t[nA]; // !!!
        calc.getChords(a, rays[idRay], grid);
        
        // Iterate over chords ...
        for(int idChord=0; idChord<nA; idChord++)
        {
#ifdef DEBUG
          if(idChannel==5)
          {
            std::cout
            << "    wait ..." << std::endl
            << "    idChannel: " << idChannel
            << "    idVoxel: " << a[idChord].getLinearIdVoxel(grid)
            << std::endl;
          }
#endif
          // Add chord length to system matrix element
          sm.setElem(idChannel, a[idChord].getLinearIdVoxel(grid),
              sm.getElem(idChannel, a[idChord].getLinearIdVoxel(grid)) + a[idChord].getLength()
          );
#ifdef DEBUG
          if(idChannel==5) std::cout << "    ok!" << std::endl;
#endif
        }
        delete[] a;
      }
      // Normalize
      for(int idVoxel=id*sm.getNVoxel(); idVoxel<(id+1)*sm.getNVoxel(); idVoxel++)
      {
        sm.setElem(idChannel, idVoxel, sm.getElem(idChannel, idVoxel)/nRays);
      }
      
      delete[] rays;
#ifdef DEBUG
      std::cout << "    channel id was: " << idChannel << ", increment now" << std::endl;
#endif
          }
    delete[] channels;
  }
};

#endif  // #ifndef SMCHUNKGETTER_HPP
