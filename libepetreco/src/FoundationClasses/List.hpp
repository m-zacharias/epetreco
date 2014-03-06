#ifndef LIST_HPP
#define LIST_HPP

// For other values than 0 make sure no one tries to access uninitialized items
#define DEFAULT_LIST_CAPACITY 0
#include <list>

/* A basic container for storing an ordered list of objects */
template<class Item>
class List : public std::list<Item>
{
  public:
    
    List( long size = DEFAULT_LIST_CAPACITY );
    
    List( List & );
    
    ~List();
    
    List & operator=( List const & );
    
    
    long count() const;
    
    Item & get( long index );

    Item & first();

    Item & last();

    bool includes( Item const & ) const;


    void append( Item const & );

    void prepend( Item const & );


//    void remove( Item const & );

    void removeLast();

    void removeFirst();

    void removeAll();


    Item & top();

    void push( Item const & );

    Item pop();
};
#include "List.tpp"

#endif  // #define LIST_HPP
