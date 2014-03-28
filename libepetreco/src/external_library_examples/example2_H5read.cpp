/* 2014-03-12
 * Example file for reading the PET measurement file. Errors for other H5 files.
 * Reads the whole 'messung' dataset and performs some basic analysis.
 */
#include "H5Cpp.h"
#include <string>
#include <iostream>

#ifndef H5_NO_NAMSPACE
using namespace H5;
#endif

int main( int ac, char ** av )
{
  if(ac!=3)
  {
    std::cout << "Error: Wrong number of arguments. Exspected 2: HDF5 file "
              << "name, dataset name" << std::endl;
  }
  
  std::string fn(av[1]);
  
  H5File * file       = new H5File(fn.c_str(), H5F_ACC_RDWR);
  DataSet dataset     = file->openDataSet(av[2]);
  
  // Get dataspace, number of dimensions, size of dimensions
  DataSpace dataspace = dataset.getSpace();
  hsize_t * dims      = new hsize_t[dataspace.getSimpleExtentNdims()];
  int ndims           = dataspace.getSimpleExtentDims(dims, NULL);
  
  // Select dataset hyperslab
  hsize_t * offset    = new hsize_t[ndims];
  for(int i=0; i<ndims; i++)
    offset[i] = 0;
  dataspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
  
  // Define memory hyperslab
  DataSpace memspace(ndims, dims);
  memspace.selectHyperslab(H5S_SELECT_SET, dims, offset);

  // Prepare memory
  int memsize = 1;
  for(int i=0; i<ndims; i++)
    memsize *= dims[i];
  float * memdata     = new float[memsize];

  // Read data from file hyperslab into memory hyperslab
  dataset.read(memdata, PredType::NATIVE_FLOAT, memspace, dataspace);
  
  // Analysis
  int nnonzero = 0;
  for(int i=0; i<memsize; i++)
    if(memdata[i]!=0)
      nnonzero++;
  std::cout << "memsize: " << memsize << std::endl;
  std::cout << "nnonzero: " << nnonzero << std::endl;
  std::cout << "fraction: " << 1.*nnonzero/memsize << std::endl;

  return 0;
}
