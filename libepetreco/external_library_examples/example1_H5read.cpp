/* 2014-03-11
 * Example file for reading the PET measurement file. Errors for other H5 files.
 * Demonstrates some basic features of reading H5 files. Follows this example:
 * http://www.hdfgroup.org/HDF5/doc/cpplus_RM/readdata_8cpp-example.html
 */
#include "H5Cpp.h"
#include <string>
#include <iostream>

int main( int argc, char ** argv )
{
  if( argc != 3 )
    std::cout << "Error: Wrong number of arguments. Exspected 2: HDF5 File name, dataset name" << std::endl;
  
  std::string fn(argv[1]);
  
  H5::H5File * file = new H5::H5File(fn.c_str(), H5F_ACC_RDWR);
  H5::DataSet dataset = file->openDataSet( argv[2] );
 
  // Get the class of the datatype that is used by the dataset in the file
  H5T_class_t type_class = dataset.getTypeClass();
  if( type_class == H5T_FLOAT )
    std::cout << "Dataset has FLOAT type" << std::endl;

  // Get the float datatype
  H5::FloatType floattype = dataset.getFloatType();

  // Get the order of the datatype
  H5std_string order_string;
  H5T_order_t order = floattype.getOrder( order_string );
  std::cout << order_string << std::endl;

  // Get the size of data elements stored in the file
  size_t size = floattype.getSize();
  std::cout << "Data element size is: " << size << std::endl;

  // Get dataspace of the dataset
  H5::DataSpace dataspace = dataset.getSpace();

  // Get the rank of the dataspace
  int rank = dataspace.getSimpleExtentNdims();
  std::cout << "Number of dimensions of dataspace is: " << rank << std::endl;

  // Get the dimension size of each dimension
  hsize_t * dims_out = new hsize_t[rank];
  int ndims = dataspace.getSimpleExtentDims( dims_out, NULL );
  std::cout << "rank: " << rank << ", dimensions: ";
  for( int i=0; i<rank; i++ )
    std::cout << (unsigned long)(dims_out[i]) << " x ";
  std::cout << std::endl;
  
  // Define hyperslab in the dataset
  hsize_t * offset = new hsize_t[rank];
  hsize_t * count = new hsize_t[rank];
  for( int i=0; i<rank; i++ )
  {
    offset[i] = 0;
    count[i] = 1;
  }
  count[0] = 180;
  dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );
  
  // Define memory dataspace
  hsize_t * dimsm = new hsize_t[1];
  dimsm[0] = 180;
  H5::DataSpace memspace(1, dimsm);
  
  // Define memory hyperslab
  hsize_t offset_out[1];
  hsize_t count_out[1];
  offset_out[0] = 0;
  count_out[0] = 180;
  memspace.selectHyperslab( H5S_SELECT_SET, count_out, offset_out );
  
  // Read data from hyperslab from file into the hyperslab in memory
  float data_out[180];
  dataset.read( data_out, H5::PredType::NATIVE_FLOAT, memspace, dataspace ); 
  
  for( int i=0; i<180; i++ )
    std::cout << i << ": " << data_out[i] << std::endl;

  return 0;
}
