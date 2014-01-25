#ifndef MATRIX_HPP
#define MATRIX_HPP

/**
 * @brief Interface to a matrix type. Each plugin has to implement a
 * specific concretisation.
 */
template<class ConcreteMatrix, typename ConcreteMatrixTraits>
class Matrix
{
  public:
    
    typedef typename ConcreteMatrixTraits::elem_t elem_t;

    /**
     * @brief Gets the number of elements in x direction ("width").
     */
    int get_nx( void )
    {
      return static_cast<ConcreteMatrix *>(this)->get_nx();
    }
    
    /**
     * @brief Gets the number of elements in y direction ("height").
     */
    int get_ny( void )
    {
      return static_cast<ConcreteMatrix *>(this)->get_ny();
    }
    
    /**
     * @brief Returns a void pointer to the raw data.
     */
    void * data( void )
    {
      return static_cast<ConcreteMatrix *>(this)->data();
    }
    
    /**
     * @brief Gets the value of a specific element of this Matrix.
     * 
     * @param idx Index of the element in x direction
     * @param idy Index of the element in y direction
     */
    elem_t get( int idx, int idy )
    {
      return static_cast<ConcreteMatrix *>(this)->get(idx, idy);
    }
    
    /**
     * @brief Sets the value of a specific element of this Matrix.
     * 
     * @param idx Index of the element in x direction
     * @param idy Index of the element in y direction
     * @param val New value of the element
     */
    void set( int idx, int idy, elem_t val )
    {
      return static_cast<ConcreteMatrix *>(this)->set(idx, idy, val);
    }
    
    /**
     * @brief Returns a pointer to a clone of this Matrix.
     */
    ConcreteMatrix * clone( void )
    {
      return static_cast<ConcreteMatrix *>(this)->clone();
    }
};

#endif  // #define MATRIX_HPP
