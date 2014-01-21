#ifndef MATRIX_HPP
#define MATRIX_HPP

/**
 * @brief Interface to a matrix type. Each plugin has to implement a
 * specific concretisation.
 */
template<class T>
class Matrix {
  public:
    /**
     * @brief Gets the number of elements in x direction ("width").
     */
    virtual int get_nx( void ) = 0;
    
    /**
     * @brief Gets the number of elements in y direction ("height").
     */
    virtual int get_ny( void ) = 0;
    
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
    virtual T get( int idx, int idy ) = 0;
    
    /**
     * @brief Sets the value of a specific element of this Matrix.
     * 
     * @param idx Index of the element in x direction
     * @param idy Index of the element in y direction
     * @param val New value of the element
     */
    virtual void set( int idx, int idy, T val ) = 0;
    
    /**
     * @brief Returns a pointer to a clone of this Matrix.
     */
    virtual Matrix * clone( void ) = 0;
};
#endif  // #define MATRIX_HPP
