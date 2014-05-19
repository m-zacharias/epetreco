#ifndef CHORDSCALC_KERNERWRAPPER
#define CHORDSCALC_KERNERWRAPPER

#include "ChordsCalc_kernel2.cu"

template<typename T, typename G, typename S>
void chordsCalc(
      int const chunkId, int const nChannels, int const chunkSize, int const nThreads,
      T * const chords,
      T * const rays,
      G * const grid,
      int const vgridSize,
      S * const setup )
{
  chordsCalc<<<chunkSize, nThreads>>>(
        chords, rays,
        grid->deviRepr()->gridO,
        grid->deviRepr()->gridD,
        grid->deviRepr()->gridN,
        chunkId*chunkSize, nChannels, chunkSize, vgridSize,
        setup->deviRepr());
}



template<typename T, typename G, typename S>
void chordsCalc_noVis(
      int const chunkId, int const nChannels, int const chunkSize, int const nThreads,
      T * const chords,
      G * const grid,
      int const vgridSize,
      S * const setup )
{
  chordsCalc_noVis<<<chunkSize, nThreads>>>(
        chords,
        grid->deviRepr()->gridO,
        grid->deviRepr()->gridD,
        grid->deviRepr()->gridN,
        chunkId*chunkSize, nChannels,
        chunkSize, vgridSize,
        setup->deviRepr());
}

#endif  // #define CHORDSCALC_KERNERWRAPPER
