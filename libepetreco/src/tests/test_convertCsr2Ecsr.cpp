#include <iostream>
#include "convertCsr2Ecsr.hpp"

int main() {
  int const ySparseRowId[] = {2, 4, 5, 9};
  int const lengthYSparseRowId(4);
  
  int const aCsrRowPtr[] = {0, 0, 0, 3, 3, 3, 4, 4, 4, 4, 6};
  int const lengthACsrRowPtr(11);
  
  int aEcsrRowPtr[5];
  
  convertCsr2Ecsr(aEcsrRowPtr, ySparseRowId, lengthYSparseRowId, aCsrRowPtr, lengthACsrRowPtr);
  
  for(int i=0; i<5; i++) {
    std::cout << aEcsrRowPtr[i] << std::endl;
  }
  
  return 0;
}
