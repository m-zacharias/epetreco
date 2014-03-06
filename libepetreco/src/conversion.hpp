#ifndef CONVERSION_HPP
#define CONVERSION_HPP

#include "cuComplex.h"

std::complex<double> convert2external( cuDoubleComplex const val )
{
  return std::complex<double>(cuCreal(val), cuCimag(val));
}

std::complex<float> convert2external( cuComplex const val )
{
  return std::complex<float>(cuCrealf(val), cuCrealf(val));
}

double convert2external( double const val )
{
  return val;
}

float convert2external( float const val )
{
  return val;
}

cuDoubleComplex convert2internal( std::complex<double> const val )
{
  return make_cuDoubleComplex(val.real(), val.imag());
}

cuComplex convert2internal( std::complex<float> const val )
{
  return make_cuComplex(val.real(), val.imag());
}

double convert2internal( double const val )
{
  return val;
}

float convert2internal( float const val )
{
  return val;
}

#endif  // #define CONVERSION_HPP
