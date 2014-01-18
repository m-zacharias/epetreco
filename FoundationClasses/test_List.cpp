#include "List.hpp"
#include <iostream>

int main()
{
  List<int> list1(5);
  List<int> list2(list1);
 
  int a = 17;

  list1.append(a);
  std::cout << list1.top() << std::endl;
  
  return 0;
}
