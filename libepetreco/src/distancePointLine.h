/** @file distancePointLine.h */
/* 
 * File:   distancePointLine.h
 * Author: malte
 *
 * Created on 17. Oktober 2014, 16:44
 */

#ifndef DISTANCEPOINTLINE_H
#define	DISTANCEPOINTLINE_H

#include <cstdio>

/**
 * Calculate absolute value of of vector.
 * @param a A 3 component array (= "vector").
 * @return Absolute value (= length) of vector.
 */
template<typename T>
__host__ __device__
inline T absolute( T const * const a )
{
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

/**
 * Calculate the scalar product of two vectors
 * @param a A 3 component array (= "vector"). One factor.
 * @param b A 3 component array (= "vector"). One factor.
 * @return Scalar product.
 */
template<typename T>
__host__ __device__
inline T scalarProduct( T const * const a, T const * const b )
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

/**
 * Calculate the minimum distance between a point and a line.
 * @param a A 3 component array (= "vector"). Position vector of one point one the line.
 * @param b A 3 component array (= "vector"). Position vector of another point on the line.
 * @param p A 3 component array (= "vector"). Position vector of the point.
 * @return Distance.
 */
template<typename T>
__host__ __device__
T distance( T const * const a, T const * const b,
                T const * const p )
{
    T ap[3];
    ap[0] = p[0]-a[0];
    ap[1] = p[1]-a[1];
    ap[2] = p[2]-a[2];
    
    T ab[3];
    ab[0] = b[0]-a[0];
    ab[1] = b[1]-a[1];
    ab[2] = b[2]-a[2];
    
    T sp = scalarProduct(ab,ap);
    T abs_ab = absolute(ab);
    T abs_ap = absolute(ap);
    
    T test = scalarProduct(ab,ap)/(absolute(ab)*absolute(ap));
    
    if(!((test>=0.) && (test<=1.))) {
      printf("test not in range! %.20e\n", test-1);
      return 0.;
    } else if(!((acos(test)>=0.) && (acos(test)<=M_PI))) {
      printf("acos(test) not in range! %f\n", acos(test));
      return 0;
    } else {
      return absolute(ap)*
              sin(acos(scalarProduct(ab,ap)/absolute(ab)/absolute(ap)));
    }
  }

#endif	/* DISTANCEPOINTLINE_H */

