#ifndef H5READER_HPP
#define H5READER_HPP

#include "H5Cpp.h"
#include <string>
#include <iostream>

class H5Reader
{
  public:

    typedef float Value_t;
    
    H5Reader( std::string filename )
    : _filename(filename), _datasetname("messung")
    {
#ifdef DEBUG
      std::cout << "H5Reader::H5Reader(std::string)" << std::endl;
#endif
    }

    void read( Value_t * mem )
    {
#ifdef DEBUG
      std::cout << "H5Reader::read(Value_t*)" << std::endl;
#endif
      // Open file, dataset
      H5::H5File * file       = new H5::H5File(_filename.c_str(), H5F_ACC_RDWR);
      H5::DataSet dataset     = file->openDataSet(_datasetname.c_str());
      
//      // Select dataset hyperslab
//      H5::DataSpace dataspace = dataset.getSpace();
//      hsize_t * dims      = new hsize_t[dataspace.getSimpleExtentNdims()];
//      int ndims = dataspace.getSimpleExtentDims(dims, NULL);
//      hsize_t * offset    = new hsize_t[ndims];
//      for(int i=0; i<ndims; i++) offset[i] = 0;
//      dataspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
//
//      // Define memory hyperslab
//      H5::DataSpace memspace(ndims, dims);
//      memspace.selectHyperslab(H5S_SELECT_SET, dims, offset);
//      
//      dataset.read(mem, H5::PredType::NATIVE_FLOAT, memspace, dataspace);
      dataset.read(mem, H5::PredType::NATIVE_FLOAT);

      delete file;
//      delete[] dims;
//      delete[] offset;
    }


  private:
    
    std::string _filename, _datasetname;
};

#endif  // #ifndef H5READER_HPP
