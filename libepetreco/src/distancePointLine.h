/* 
 * File:   distancePointLine.h
 * Author: malte
 *
 * Created on 17. Oktober 2014, 16:44
 */

#ifndef DISTANCEPOINTLINE_H
#define	DISTANCEPOINTLINE_H

#include <iomanip>

template<typename T>
inline T absolute( T const * const a )
{
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

template<typename T>
inline T scalarProduct( T const * const a, T const * const b )
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

template<typename T>
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
      std::cerr << std::scientific << std::setw(20) << "test not in range! " << test-1. << std::endl;
      return 0.;
    } else if(!((acos(test)>=0.) && (acos(test)<=M_PI))) {
      std::cerr << "acos(test) not in range! " << acos(test) << std::endl;
      return 0;
    } else {
      return absolute(ap)*
              sin(acos(scalarProduct(ab,ap)/absolute(ab)/absolute(ap)));
    }
  }

//template<typename T>
//T distance( T const * const a, T const * const b,
//                T const * const p )
//{
//    T ap[3];
//    ap[0] = p[0]-a[0];
//    ap[1] = p[1]-a[1];
//    ap[2] = p[2]-a[2];
//    
//    T ab[3];
//    ab[0] = b[0]-a[0];
//    ab[1] = b[1]-a[1];
//    ab[2] = b[2]-a[2];
//    
//    T x = scalarProduct(ab,ap)/absolute(ab)/absolute(ap);
//    
//    return absolute(ap)*sqrt(1-(x*x));
//}

#endif	/* DISTANCEPOINTLINE_H */

