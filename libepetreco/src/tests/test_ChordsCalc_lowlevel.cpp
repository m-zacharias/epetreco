#include "ChordsCalc_lowlevel.hpp"

typedef float val_t;
int main()
{
  val_t const ray[6] = {-0.5, 1.0, 0.4, 3.5, 1.0, 0.6};
  val_t const gridO[3] = {0.0, 0.0, 0.0};
  val_t const gridD[3] = {1.0, 1.0, 1.0};
  int   const gridN[3] = {3,   3,   3};

  val_t chords[5] = {0,0,0,0,0};
  int voxelIds[15];

  getChords(chords, voxelIds, 3, ray, gridO, gridD, gridN);

  return 0;
}
