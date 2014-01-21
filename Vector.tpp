#include "Vector.hpp"

template<class T>
Vector<T>::Vector( T const x_, T const y_, T const z_ )
: x(x_), y(y_), z(z_) {}

template<class T>
Vector<T>::Vector( Vector const & v )
: x(v.x), y(v.y), z(v.z) {}

template<class T>
void Vector<T>::operator=( Vector<T> const & v )
{
  x=v.x; y=v.y; z=v.z;
}
