#include "TemplateVertex.hpp"

template<class T>
TemplateVertex<T>::TemplateVertex( T const x_, T const y_, T const z_ )
: x(x_), y(y_), z(z_) {}

template<class T>
TemplateVertex<T>::TemplateVertex( TemplateVertex const & v )
: x(v.x), y(v.y), z(v.z) {}

template<class T>
void TemplateVertex<T>::operator=( TemplateVertex<T> const & v )
{
  x=v.x; y=v.y; z=v.z;
}
