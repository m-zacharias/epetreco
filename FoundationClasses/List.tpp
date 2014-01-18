#include "List.hpp"

#ifdef DEBUG
#include <iostream>
#include <typeinfo>
#endif

template<class Item>
List<Item>::List( long size )
: std::list<Item>(size)
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name() << ">::List(long)" << std::endl;
#endif
}

template<class Item>
List<Item>::List( List & list )
: std::list<Item>(list)
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name() << ">::List(List&)" << std::endl;
#endif
}

template<class Item>
List<Item>::~List() 
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name() << ">::~List()" << std::endl;
#endif
}

template<class Item>
List<Item> & List<Item>::operator=( List<Item> const & list )
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::operator=(List<Item=" << typeid(Item).name() 
            << "> const &)" << std::endl;
#endif
  this->std::list<Item>::operator=(list);
}


template<class Item>
long List<Item>::count() const
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::count() const" << std::endl;
#endif
  return static_cast<long>(this->std::list<Item>::size());
}

template<class Item>
Item & List<Item>::get( long index )
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::get(long)" << std::endl;
#endif
  typename std::list<Item>::iterator it = this->begin();
  long i = 0;
  while(i<index && it!=this->end()) {
    it++;
    i++;
  }
  return *it;
}

template<class Item>
Item & List<Item>::first()
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::first()" << std::endl;
#endif
  return std::list<Item>::front();
}

template<class Item>
Item & List<Item>::last()
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::last()" << std::endl;
#endif
  return std::list<Item>::back();
}

template<class Item>
bool List<Item>::includes( Item const & item) const
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::includes(" << typeid(Item).name()
            << " const &) const" << std::endl;
#endif
  typename std::list<Item>::iterator it = this->begin();
  bool includes = false;
  while(it!=this->end()) {
    includes = includes || *it==item;
    it++;
  }
  return includes;
}


template<class Item>
void List<Item>::append( Item const & item )
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::append(" << typeid(Item).name()
            << " const &)" << std::endl;
#endif
  std::list<Item>::push_back(item);
}

template<class Item>
void List<Item>::prepend( Item const & item )
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::prepend(" << typeid(Item).name()
            << " const &)" << std::endl;
#endif
  std::list<Item>::push_front(item);
}


template<class Item>
void List<Item>::removeLast()
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::removeLast()" << std::endl;
#endif
  std::list<Item>::pop_back();
}

template<class Item>
void List<Item>::removeFirst()
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::removeFirst()" << std::endl;
#endif
  std::list<Item>::pop_front();
}

template<class Item>
void List<Item>::removeAll()
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::removeAll()" << std::endl;
#endif
  std::list<Item>::clear();
}


template<class Item>
Item & List<Item>::top()
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::top()" << std::endl;
#endif
  return last();
}

template<class Item>
void List<Item>::push( Item const & item )
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::push(" << typeid(Item).name()
            << " const &)" << std::endl;
#endif
  append(item);
}

template<class Item>
Item List<Item>::pop()
{
#ifdef DEBUG
  std::cout << "List<Item=" << typeid(Item).name()
            << ">::pop()" << std::endl;
#endif
  Item pop = last();
  removeLast();
  return pop;
}
