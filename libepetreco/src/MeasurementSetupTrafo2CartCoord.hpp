/* 
 * File:   MeasurementSetupTrafo2PixelVol.hpp
 * Author: malte
 *
 * Created on 14. Oktober 2014, 18:23
 */

/*
 * TODO:
 * - put transformation matrix calculation in constructor
 * - reuse transformation matrix -> calculate it only once in construction
 * - put argument "meas" in Ctor (?), so that functors are created for one
 *   specific measurement setup only
 * - asserts: are arguments within valid range?
 */

#ifndef MEASUREMENTSETUPTRAFO2CARTCOORD_HPP
#define	MEASUREMENTSETUPTRAFO2CARTCOORD_HPP

#include "MeasurementSetup.hpp"
#include <cmath>

template<
      typename T
    , typename ConcreteMSTrafo2CartCoordFirstPixel
    , typename ConcreteMS >
class MeasurementSetupTrafo2CartCoordFirstPixel {
  public:
    __host__ __device__
    void operator()(
          T * const cart_coords, T const * const box_coords,
          int const id0z, int const id0y, int const ida,
          ConcreteMS const * const meas ) {
      return static_cast<ConcreteMSTrafo2CartCoordFirstPixel*>(this)->
              operator()(cart_coords, box_coords,
                         id0z, id0y, ida,
                         meas);
    }
};

template<
      typename T
    , typename ConcreteMS >
class DefaultMeasurementSetupTrafo2CartCoordFirstPixel
: public MeasurementSetupTrafo2CartCoordFirstPixel<
             T,
             DefaultMeasurementSetupTrafo2CartCoordFirstPixel<T, ConcreteMS>,
             ConcreteMS> {
  public:
    __host__ __device__
    void operator()(
          T * const cart_coords, T const * const box_coords,
          int const id0z, int const id0y, int const ida,
          ConcreteMS const * const meas ) {
      // get pixel edge leghts
      T edges[3];
      edges[0] = meas->segx();
      edges[1] = meas->segy();
      edges[2] = meas->segz();
      // get not-yet-rotated position of pixel
      T pos[3];
      pos[0]=meas->pos0x();
      pos[1]=(id0y-.5*(meas->n0y()-1))*edges[1];
      pos[2]=(id0z-.5*(meas->n0z()-1))*edges[2];
      // get angular function values of channel's rotation
      T sin_ = sin(ida*(meas->da()));
      T cos_ = cos(ida*(meas->da()));

      // create transformation matrix
      T trafo[12];

      trafo[0*4 + 0] = cos_*edges[0];
      trafo[0*4 + 1] = 0.;
      trafo[0*4 + 2] = sin_*edges[2];
      trafo[0*4 + 3] = cos_*(pos[0]-.5*edges[0])\
                      +sin_*(pos[2]-.5*edges[2]);

      trafo[1*4 + 0] = 0.;
      trafo[1*4 + 1] = edges[1];
      trafo[1*4 + 2] = 0.;
      trafo[1*4 + 3] = pos[1]-.5*edges[1];

      trafo[2*4 + 0] =-sin_*edges[0];
      trafo[2*4 + 1] = 0.;
      trafo[2*4 + 2] = cos_*edges[2];
      trafo[2*4 + 3] =-sin_*(pos[0]-.5*edges[0])\
                      +cos_*(pos[2]-.5*edges[2]);

      // apply transformation matrix
      cart_coords[0] =   trafo[0*4 + 0] * box_coords[0]
                       + trafo[0*4 + 1] * box_coords[1]
                       + trafo[0*4 + 2] * box_coords[2] 
                       + trafo[0*4 + 3] * 1.;

      cart_coords[1] =   trafo[1*4 + 0] * box_coords[0]
                       + trafo[1*4 + 1] * box_coords[1]
                       + trafo[1*4 + 2] * box_coords[2] 
                       + trafo[1*4 + 3] * 1.;

      cart_coords[2] =   trafo[2*4 + 0] * box_coords[0]
                       + trafo[2*4 + 1] * box_coords[1]
                       + trafo[2*4 + 2] * box_coords[2] 
                       + trafo[2*4 + 3] * 1.;
    }
};

template<
      typename T
    , typename ConcreteMSTrafo2CartCoordSecndPixel
    , typename ConcreteMS >
class MeasurementSetupTrafo2CartCoordSecndPixel {
  public:
    __host__ __device__
    void operator()(
          T * const cart_coords, T const * const box_coords,
          int const id1z, int const id1y, int const ida,
          ConcreteMS const * const meas ) {
      return static_cast<ConcreteMSTrafo2CartCoordSecndPixel*>(this)->
              operator()(cart_coords, box_coords,
                         id1z, id1y, ida,
                         meas);
    }
};

template<
      typename T
    , typename ConcreteMS >
class DefaultMeasurementSetupTrafo2CartCoordSecndPixel
: public MeasurementSetupTrafo2CartCoordSecndPixel<
             T,
             DefaultMeasurementSetupTrafo2CartCoordSecndPixel<T, ConcreteMS>,
             ConcreteMS> {
  public:
    __host__ __device__
    void operator()(
          T * const cart_coords, T const * const box_coords,
          int const id1z, int const id1y, int const ida,
          ConcreteMS const * const meas ) {
      // get pixel edge leghts
      T edges[3];
      edges[0] = meas->segx();
      edges[1] = meas->segy();
      edges[2] = meas->segz();
      // get not-yet-rotated position of pixel
      T pos[3];
      pos[0]=meas->pos1x();
      pos[1]=(id1y-.5*(meas->n1y()-1))*edges[1];
      pos[2]=(id1z-.5*(meas->n1z()-1))*edges[2];
      // get angular function values of channel's rotation
      T sin_ = sin(ida*(meas->da()));
      T cos_ = cos(ida*(meas->da()));

      // create transformation matrix
      T trafo[12];

      trafo[0*4 + 0] = cos_*edges[0];
      trafo[0*4 + 1] = 0.;
      trafo[0*4 + 2] = sin_*edges[2];
      trafo[0*4 + 3] = cos_*(pos[0]-.5*edges[0])\
                      +sin_*(pos[2]-.5*edges[2]);

      trafo[1*4 + 0] = 0.;
      trafo[1*4 + 1] = edges[1];
      trafo[1*4 + 2] = 0.;
      trafo[1*4 + 3] = pos[1]-.5*edges[1];

      trafo[2*4 + 0] =-sin_*edges[0];
      trafo[2*4 + 1] = 0.;
      trafo[2*4 + 2] = cos_*edges[2];
      trafo[2*4 + 3] =-sin_*(pos[0]-.5*edges[0])\
                      +cos_*(pos[2]-.5*edges[2]);

      // apply transformation matrix
      cart_coords[0] =   trafo[0*4 + 0] * box_coords[0]
                       + trafo[0*4 + 1] * box_coords[1]
                       + trafo[0*4 + 2] * box_coords[2] 
                       + trafo[0*4 + 3] * 1.;

      cart_coords[1] =   trafo[1*4 + 0] * box_coords[0]
                       + trafo[1*4 + 1] * box_coords[1]
                       + trafo[1*4 + 2] * box_coords[2] 
                       + trafo[1*4 + 3] * 1.;

      cart_coords[2] =   trafo[2*4 + 0] * box_coords[0]
                       + trafo[2*4 + 1] * box_coords[1]
                       + trafo[2*4 + 2] * box_coords[2] 
                       + trafo[2*4 + 3] * 1.;
    }
};
#endif	/* MEASUREMENTSETUPTRAFO2CARTCOORD_HPP */

