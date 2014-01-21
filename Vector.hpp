#ifndef VECTOR_HPP
#define VECTOR_HPP

template<typename T>
struct Vector {

  public:
    
    /* Parametric Constructor */
    Vector( T const x_, T const y_, T const z_ );
    
    /* Copy Constructor */
    Vector( Vector const & v );

    /* Copy Assignment */
    void operator=( Vector const & v );
    


    T x, y, z;
};
#include "Vector.tpp"

#endif  // #define VECTOR_HPP
