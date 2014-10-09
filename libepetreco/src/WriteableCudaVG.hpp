#ifndef WRITEABLECUDAVG
#define WRITEABLECUDAVG

#include "CudaVG.hpp"

template<typename T, typename ConcreteVoxelGrid>
class WriteableCudaVG : public CudaVG<T, ConcreteVoxelGrid>
{
  public:

    WriteableCudaVG(
          T const   gridO0, T const   gridO1, T const   gridO2,
          T const   gridD0, T const   gridD1, T const   gridD2,
          int const gridN0, int const gridN1, int const gridN2 )
    : CudaVG<T, ConcreteVoxelGrid>(
          gridO0, gridO1, gridO2,
          gridD0, gridD1, gridD2,
          gridN0, gridN1, gridN2) {}

    void getOrigin( float * origin )
    {
      for(int dim=0; dim<3; dim++)
        origin[dim] = this->hostRepr()->gridO[dim];
    }

    void getVoxelSize( float * voxelSize )
    {
      for(int dim=0; dim<3; dim++)
        voxelSize[dim] = this->hostRepr()->gridD[dim];
    }

    void getNumberOfVoxels( int * numberOfVoxels )
    {
      for(int dim=0; dim<3; dim++)
        numberOfVoxels[dim] = this->hostRepr()->gridN[dim];
    }
};

#endif  // #ifndef WRITEABLECUDAVG

