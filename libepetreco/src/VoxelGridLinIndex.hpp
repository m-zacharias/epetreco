/* 
 * File:   VoxelGridLinIndex.hpp
 * Author: malte
 *
 * Created on 22. Oktober 2014, 13:15
 */

#ifndef VOXELGRIDLININDEX_HPP
#define	VOXELGRIDLININDEX_HPP

template<
      typename ConcreteVGLinId
    , typename ConcreteVG >
struct VoxelGridLinId {
  __host__ __device__
  int operator()(
        int const idx, int const idy, int const idz,
        ConcreteVG const * const vg ) {
    return static_cast<ConcreteVG*>(this)->
      operator()(idx, idy, idz, vg);
  }
};

template<
      typename ConcreteVG >
struct DefaultVoxelGridLinId
: public VoxelGridLinId<DefaultVoxelGridLinId<ConcreteVG>, ConcreteVG> {
  __host__ __device__
  int operator()(
        int const idx, int const idy, int const idz,
        ConcreteVG const * const vg ) {
    return   idx
           + idy * vg->gridnx()
           + idz *(vg->gridnx()*vg->gridny());
  }
};

template<
      typename ConcreteIdx,
      typename ConcreteVG >
struct VoxelGridIdx {
  __host__ __device__
  int operator()(
        int const linId,
        ConcreteVG const * const vg ) {
    return static_cast<ConcreteVG*>(this)->
      operator()(linId, vg);
  }
};

template<
      typename ConcreteVG >
struct DefaultVoxelGridIdx
: public VoxelGridIdx<DefaultVoxelGridIdx<ConcreteVG>, ConcreteVG> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteVG const * const vg ) {
    int temp(linId);
    temp %= (vg->gridnx()*vg->gridny());
    temp %=  vg->gridnx();
    return temp/1;
  }
};

template<
      typename ConcreteIdy,
      typename ConcreteVG >
struct VoxelGridIdy {
  __host__ __device__
  int operator()(
        int const linId,
        ConcreteVG const * const vg ) {
    return static_cast<ConcreteVG*>(this)->
      operator()(linId, vg);
  }
};

template<
      typename ConcreteVG >
struct DefaultVoxelGridIdy
: public VoxelGridIdy<DefaultVoxelGridIdy<ConcreteVG>, ConcreteVG> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteVG const * const vg ) {
    int temp(linId);
    temp %= (vg->gridnx()*vg->gridny());
    return temp/vg->gridnx();
  }
};

template<
      typename ConcreteIdz,
      typename ConcreteVG >
struct VoxelGridIdz {
  __host__ __device__
  int operator()(
        int const linId,
        ConcreteVG const * const vg ) {
    return static_cast<ConcreteVG*>(this)->
      operator()(linId, vg);
  }
};

template<
      typename ConcreteVG >
struct DefaultVoxelGridIdz
: public VoxelGridIdz<DefaultVoxelGridIdz<ConcreteVG>, ConcreteVG> {
  __host__ __device__
  int operator()(
        int const linId, ConcreteVG const * const vg ) {
    return linId/(vg->gridnx()*vg->gridny());
  }
};
#endif	/* VOXELGRIDLININDEX_HPP */

