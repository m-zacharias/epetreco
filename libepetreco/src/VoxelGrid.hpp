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
    
    T gridox() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridox();
    }

    T gridoy() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridoy();
    }

    T gridoz() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridoz();
    }
    
    T griddx() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->griddx();
    }

    T griddy() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->griddy();
    }

    T griddz() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->griddz();
    }
    
    int gridnx() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridnx();
    }

    int gridny() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridny();
    }

    int gridnz() const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->gridnz();
    }

    int linVoxelId( int const * const sepVoxelId ) const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->linVoxelId(sepVoxelId);
    }
    
    void sepVoxelId( int * const sepVoxelId, int const linVoxelId ) const
    {
        return static_cast<ConcreteVoxelGrid*>(this)->
                sepVoxelId(sepVoxelId, linVoxelId);
    }
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
    
    T gridox() const
    {
        return gridO[0];
    }
    
    T gridoy() const
    {
        return gridO[1];
    }
    
    T gridoz() const
    {
        return gridO[2];
    }
    
    T griddx() const
    {
        return gridD[0];
    }
    
    T griddy() const
    {
        return gridD[1];
    }
    
    T griddz() const
    {
        return gridD[2];
    }
    
    int gridnx() const
    {
        return gridN[0];
    }
    
    int gridny() const
    {
        return gridN[1];
    }
    
    int gridnz() const
    {
        return gridN[2];
    }
    
    int linVoxelId( int const * const sepVoxelId ) const
    {
        return   sepVoxelId[0]
               + sepVoxelId[1]*gridN[0]
               + sepVoxelId[2]*gridN[0]*gridN[1];
    }
    
    void sepVoxelId( int * const sepVoxelId, int const linVoxelId ) const
    {
        int tmp = linVoxelId;
        sepVoxelId[2] = tmp/gridN[1]/gridN[0];
        tmp = tmp%(gridN[1]*gridN[0]);
        sepVoxelId[1] = tmp/gridN[0];
        tmp = tmp%gridN[0];
        sepVoxelId[2] = tmp;
    }
};

#endif  // #define VOXELGRID_HPP
