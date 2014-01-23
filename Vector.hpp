#ifndef VECTOR_HPP
#define VECTOR_HPP

/**
 * @brief Interface to a vector type. Each plugin has to implement a
 * specific concretisation.
 */
template<class T>
class Vector
{
  public:
    
    typedef T external_elem_t;

    /**
     * @brief Gets the number of elements in x direction ("width").
     */
    virtual int get_n( void ) = 0;
    
    /**
     * @brief Returns a void pointer to the raw data.
     */
    virtual void * data( void ) = 0;
    
    /**
     * @brief Gets the value of a specific element of this Matrix.
     * 
     * @param idx Index of the element in x direction
     * @param idy Index of the element in y direction
     */
    virtual T get( int id ) = 0;
    
    /**
     * @brief Sets the value of a specific element of this Matrix.
     * 
     * @param idx Index of the element in x direction
     * @param idy Index of the element in y direction
     * @param val New value of the element
     */
    virtual void set( int id, T val ) = 0;
    
    /**
     * @brief Returns a pointer to a clone of this Matrix.
     */
    virtual Vector * clone( void ) = 0;
};

#endif  // #define VECTOR_HPP
