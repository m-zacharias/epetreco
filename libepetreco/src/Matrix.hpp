/** @file Matrix.hpp */
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
     * @brief Gets the number of columns ("width").
     */
    int getNCols( void )
    {
      return static_cast<ConcreteMatrix *>(this)->getNCols();
    }
    
    /**
     * @brief Gets the number of rows ("height").
     */
    int getNRows( void )
    {
      return static_cast<ConcreteMatrix *>(this)->getNRows();
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
     * @param colId Index of column
     * @param rowId Index of row
     */
    elem_t get( int rowId, int colId )
    {
      return static_cast<ConcreteMatrix *>(this)->get(rowId, colId);
    }
    
    /**
     * @brief Sets the value of a specific element of this Matrix.
     * 
     * @param rowId Index of column
     * @param colId Index of row
     * @param val New value of the element
     */
    void set( int rowId, int colId, elem_t val )
    {
      return static_cast<ConcreteMatrix *>(this)->set(rowId, colId, val);
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
