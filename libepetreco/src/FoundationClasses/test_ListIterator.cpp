#include "ListIterator.hpp"
#include "List.hpp"
#include <iostream>

int main( void )
{
  List<int> list(0);
  list.append(1);
  list.append(2);
  list.append(3);
  list.append(4);

  ListIterator<int> it(&list);
  for(it.first(); !it.isDone(); it.next()) {
    std::cout << "Current list item is: " << it.currentItem() << std::endl;
  }

  return 0;
}
