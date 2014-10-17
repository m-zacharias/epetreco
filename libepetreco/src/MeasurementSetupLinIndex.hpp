/* 
 * File:   MeasurementSetupLinIndex.hpp
 * Author: malte
 *
 * Created on 12. Oktober 2014, 16:01
 */

/*
 * TODO:
 * - put argument "meas" in Ctor (?), so that functors are created for one
 *   specific measurement setup only
 * - asserts: are arguments within valid range?
 */

#ifndef MEASUREMENTSETUPLININDEX_HPP
#define	MEASUREMENTSETUPLININDEX_HPP

#include "MeasurementSetup.hpp"

template<typename ConcreteMSLinId, typename ConcreteMeasurementSetup>
struct MeasurementSetupLinId {
  int operator()(int const id0z, int const id0y, int const id1z, int const id1y,
                 int const ida, ConcreteMeasurementSetup const * const meas) {
    return static_cast<ConcreteMSLinId*>(this)->
            operator()(id0z, id0y, id1z, id1y, ida, meas);
  }
};

template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupLinId
: public MeasurementSetupLinId<DefaultMeasurementSetupLinId<ConcreteMeasurementSetup>,
                                ConcreteMeasurementSetup> {
  int operator()(int const id0z, int const id0y, int const id1z, int const id1y,
                 int const ida, ConcreteMeasurementSetup const * const meas) {
    return   id0z
           + id0y * (meas->n0z())
           + id1z * (meas->n0z())*(meas->n0y())
           + id1y * (meas->n0z())*(meas->n0y())*(meas->n1z())
           + ida  * (meas->n0z())*(meas->n0y())*(meas->n1z())*(meas->n1y());
  }
};

template<typename ConcreteMSId0z, typename ConcreteMeasurementSetup>
struct MeasurementSetupId0z {
  int operator()(int const linId, ConcreteMeasurementSetup const * const meas) {
    return static_cast<ConcreteMSId0z*>(this)->
            operator()(linId, meas);
  }
};

template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupId0z
: public MeasurementSetupId0z<DefaultMeasurementSetupId0z<ConcreteMeasurementSetup>,
                                ConcreteMeasurementSetup> {
  int operator()(int const linId, ConcreteMeasurementSetup const * const meas) {
    int temp = linId;
    temp %=      ((meas->n0z())*(meas->n0y())*(meas->n1z())*(meas->n1y()));
    temp %=      ((meas->n0z())*(meas->n0y())*(meas->n1z()));
    temp %=      ((meas->n0z())*(meas->n0y()));
    temp %=      ((meas->n0z()));
    return temp;
  }
};

template<typename ConcreteMSId0y, typename ConcreteMeasurementSetup>
struct MeasurementSetupId0y {
  int operator()(int const linId, ConcreteMeasurementSetup const * const meas) {
    return static_cast<ConcreteMSId0y*>(this)->
            operator()(linId, meas);
  }
};

template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupId0y
: public MeasurementSetupId0y<DefaultMeasurementSetupId0y<ConcreteMeasurementSetup>,
                                ConcreteMeasurementSetup> {
  int operator()(int const linId, ConcreteMeasurementSetup const * const meas) {
    int temp = linId;
    temp %=      ((meas->n0z())*(meas->n0y())*(meas->n1z())*(meas->n1y()));
    temp %=      ((meas->n0z())*(meas->n0y())*(meas->n1z()));
    temp %=      ((meas->n0z())*(meas->n0y()));
    return temp /((meas->n0z()));
  }
};

template<typename ConcreteMSId1z, typename ConcreteMeasurementSetup>
struct MeasurementSetupId1z {
  int operator()(int const linId, ConcreteMeasurementSetup const * const meas) {
    return static_cast<ConcreteMSId1z*>(this)->
            operator()(linId, meas);
  }
};

template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupId1z
: public MeasurementSetupId1z<DefaultMeasurementSetupId1z<ConcreteMeasurementSetup>,
                                ConcreteMeasurementSetup> {
  int operator()(int const linId, ConcreteMeasurementSetup const * const meas) {
    int temp = linId;
    temp %=      ((meas->n0z())*(meas->n0y())*(meas->n1z())*(meas->n1y()));
    temp %=      ((meas->n0z())*(meas->n0y())*(meas->n1z()));
    return temp /((meas->n0z())*(meas->n0y()));
  }
};

template<typename ConcreteMSId1y, typename ConcreteMeasurementSetup>
struct MeasurementSetupId1y {
  int operator()(int const linId, ConcreteMeasurementSetup const * const meas) {
    return static_cast<ConcreteMSId1y*>(this)->
            operator()(linId, meas);
  }
};

template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupId1y
: public MeasurementSetupId1y<DefaultMeasurementSetupId1y<ConcreteMeasurementSetup>,
                                ConcreteMeasurementSetup> {
  int operator()(int const linId, ConcreteMeasurementSetup const * const meas) {
    int temp = linId;
    temp %=      ((meas->n0z())*(meas->n0y())*(meas->n1z())*(meas->n1y()));
    return temp /((meas->n0z())*(meas->n0y())*(meas->n1z()));
  }
};

template<typename ConcreteMSIda, typename ConcreteMeasurementSetup>
struct MeasurementSetupIda {
  int operator()(int const linId, ConcreteMeasurementSetup const * const meas) {
    return static_cast<ConcreteMSIda*>(this)->
            operator()(linId, meas);
  }
};

template<typename ConcreteMeasurementSetup>
struct DefaultMeasurementSetupIda
: public MeasurementSetupIda<DefaultMeasurementSetupIda<ConcreteMeasurementSetup>,
                              ConcreteMeasurementSetup> {
  int operator()(int const linId, ConcreteMeasurementSetup const * const meas) {
    return linId /((meas->n0z())*(meas->n0y())*(meas->n1z())*(meas->n1y()));
  }
};

#endif	/* MEASUREMENTSETUPLININDEX_HPP */

