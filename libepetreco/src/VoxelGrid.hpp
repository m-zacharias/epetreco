#ifndef VOXELGRID_HPP
#define VOXELGRID_HPP

template<typename T, typename ConcreteVoxelGrid>
class VoxelGrid
{
  public:
    
    VoxelGrid( T const   gridO0, T const   gridO1, T const   gridO2,
               T const   gridD0, T const   gridD1, T const   gridD2,
               int const gridN0, int const gridN1, int const gridN2 )
    {}
    
//    __host__ __device__
//    void getGridO( T * const gridO ) const
//    {
//      return static_cast<ConcreteVoxelGrid*>(this)->getGridO(gridO);
//    }
//    
//    __host__ __device__
//    void getGridD( T * const gridD ) const
//    {
//      return static_cast<ConcreteVoxelGrid*>(this)->getGridD(gridD);
//    }
//    
//    __host__ __device__
//    void getGridN( int * const gridN ) const
//    {
//      return static_cast<ConcreteVoxelGrid*>(this)->getGridN(gridN);
//    }
};



template<typename T>
class DefaultVoxelGrid : public VoxelGrid<T, DefaultVoxelGrid<T> >
{
  public:
    
    DefaultVoxelGrid( T const   gridO0, T const   gridO1, T const   gridO2,
                      T const   gridD0, T const   gridD1, T const   gridD2,
                      int const gridN0, int const gridN1, int const gridN2 )
    : VoxelGrid<T, DefaultVoxelGrid<T> >(gridO0, gridO1, gridO2,
                                         gridD0, gridD1, gridD2,  
                                         gridN0, gridN1, gridN2)
    {
      gridO[0]=gridO0; gridO[1]=gridO1; gridO[2]=gridO2;
      gridD[0]=gridD0; gridD[1]=gridD1; gridD[2]=gridD2;
      gridN[0]=gridN0; gridN[1]=gridN1; gridN[2]=gridN2;
    }

    T   gridO[3];
    T   gridD[3];
    int gridN[3];
    
//    __host__ __device__
//    void getGridO( T * const gridO ) const
//    {
//      for(int dim=0; dim<3; dim++) gridO[dim] = _gridO[dim];
//    }
//    
//    __host__ __device__
//    void getGridD( T * const gridD ) const
//    {
//      for(int dim=0; dim<3; dim++) gridD[dim] = _gridD[dim];
//    }
//    
//    __host__ __device__
//    void getGridN( int * const gridN ) const
//    {
//      for(int dim=0; dim<3; dim++) gridN[dim] = _gridN[dim];
//    }
};

#endif  // #define VOXELGRID_HPP
