/* TODO:
 * For specific List implementation that uses std::list, ListIterator should
 * accordingly be implemented using std::list::iterator. */

#include "ListIterator.hpp"

#ifdef DEBUG_FOUNDATIONCLASSES
#include <iostream>
#include <typeinfo>
#endif

template<class Item>
ListIterator<Item>::ListIterator( List<Item> * list )
: _list(list)
{
#ifdef DEBUG_FOUNDATIONCLASSES
  std::cout << "ListIterator<Item=" << typeid(Item).name()
            << ">::ListIterator()"  << std::endl;
#endif
}

template<class Item>
void ListIterator<Item>::first()
{
#ifdef DEBUG_FOUNDATIONCLASSES
  std::cout << "ListIterator<Item=" << typeid(Item).name()
            << ">::first()"         << std::endl;
#endif
  _position = 0;
}

template<class Item>
void ListIterator<Item>::next()
{
#ifdef DEBUG_FOUNDATIONCLASSES
  std::cout << "ListIterator<Item=" << typeid(Item).name()
            << ">::next()"          << std::endl;
#endif
  _position++;
}

template<class Item>
bool ListIterator<Item>::isDone() const
{
#ifdef DEBUG_FOUNDATIONCLASSES
  std::cout << "ListIterator<Item=" << typeid(Item).name()
            << ">::isDone()"        << std::endl;
#endif
  return _position >= _list->count();
}

template<class Item>
Item & ListIterator<Item>::currentItem()
{
#ifdef DEBUG_FOUNDATIONCLASSES
  std::cout << "ListIterator<Item=" << typeid(Item).name()
            << ">::currentItem()"   << std::endl;
#endif
  return _list->get(_position);
}
