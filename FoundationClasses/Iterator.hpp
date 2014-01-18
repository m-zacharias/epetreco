#ifndef ITERATOR_HPP
#define ITERATOR_HPP

/* Abstract class that defines a traversal interface for aggregates */
template<class Item>
class Iterator
{
  public:
    
    /* Position iterator to the first object in the aggregate */
    virtual void first() = 0;
    
    /* Position iterator to the next object in the sequence */
    virtual void next() = 0;
    
    /* Returns true when therer are no more objects in the sequence */
    virtual bool isDone() const = 0;
    
    /* Returns the object at the current position in the sequence */
    virtual Item & currentItem() = 0;


  protected:
    
    /* Protected Constructor prevents instanciation */
    Iterator()
    {}
};

#endif  // #define ITERATOR_HPP
