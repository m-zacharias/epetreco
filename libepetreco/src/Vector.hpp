#ifndef VECTOR_HPP
#define VECTOR_HPP

/**
 * @brief Interface to a vector type. Each plugin has to implement a
 * specific concretisation.
 */
template<class ConcreteVector, typename ConcreteVectorTraits>
class Vector
{
  public:
    
    typedef typename ConcreteVectorTraits::elem_t elem_t;

    /**
     * @brief Gets the number of elements in x direction ("width").
     */
    int getN( void )
    {
      return static_cast<ConcreteVector *>(this)->getN();
    }
    
    /**
     * @brief Returns a void pointer to the raw data.
     */
    void * data( void )
    {
      return static_cast<ConcreteVector *>(this)->data();
    }
    
    /**
     * @brief Gets the value of a specific element of this Matrix.
     * 
     * @param idx Index of the element in x direction
     * @param idy Index of the element in y direction
     */
    elem_t get( int id )
    {
      return static_cast<ConcreteVector *>(this)->get(id);
    }
    
    /**
     * @brief Sets the value of a specific element of this Matrix.
     * 
     * @param idx Index of the element in x direction
     * @param idy Index of the element in y direction
     * @param val New value of the element
     */
    void set( int id, elem_t val )
    {
      static_cast<ConcreteVector *>(this)->set(id, val);
    }
    
    /**
     * @brief Returns a pointer to a clone of this Matrix.
     */
    ConcreteVector * clone( void )
    {
      return static_cast<ConcreteVector *>(this)->clone();
    }
};

#endif  // #define VECTOR_HPP
