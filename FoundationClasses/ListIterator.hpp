#ifndef LISTITERATOR_HPP
#define LISTITERATOR_HPP

#include "Iterator.hpp"
#include "List.hpp"

template<class Item>
class ListIterator : public Iterator<Item>
{
  public:
    
    ListIterator( List<Item> * aList );

    virtual void first();

    virtual void next();

    virtual bool isDone() const;

    virtual Item & currentItem();


  protected:
    
    List<Item> * _list;
    long _position;
};
#include "ListIterator.tpp"

#endif  // #define LISTITERATOR_HPP
