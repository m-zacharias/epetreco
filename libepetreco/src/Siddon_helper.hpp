#ifndef SIDDON_HELPER_HPP
#define SIDDON_HELPER_HPP

#include <algorithm>

/* Minimum of three numbers */
/**
 * @param a Candidate for minimum
 * @param b Candidate for minimum
 * @param c Candidate for minimum
 */
template<typename T>
T min( T a, T b, T c )
{
  return std::min(std::min(a, b), c);
}


/* Conditional minimum of three numbers */
/**
 * @param a Possibly a canditate for minimum
 * @param b Possibly a canditate for minimum
 * @param c Possibly a canditate for minimum
 * @param ab a actually is a canditate: true/false
 * @param bb b actually is a canditate: true/false
 * @param cb c actually is a canditate: true/false
 */
template<typename T>
T min( T a, T b, T c, bool ab, bool bb, bool cb )
{
  if( !ab ) {
    if( !bb ) {
      if( !cb ) {
        throw -1;
      }
      else {
        return c;
      }
    }
    else {
      if( !cb ) {
        return b;
      }
      else {
        return std::min(b, c);
      }
    }
  }
  else {
    if( !bb ) {
      if( !cb ) {
        return a;
      }
      else {
        return std::min(a, c);
      }
    }
    else {
      if( !cb ) {
        return std::min(a, b);
      }
      else {
        return std::min(std::min(a, b), c);
      }
    }
  }
}

#endif  // #ifndef SIDDON_HELPER_HPP
