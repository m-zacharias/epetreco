/** @file H5DensityReader.hpp */
/* Author: malte
 *
 * Created on 9. April 2015, 15:14 */

#ifndef H5DENSITYREADER_HPP
#define	H5DENSITYREADER_HPP

#include "H5Cpp.h"

class H5DensityReader {
public:
  typedef float Value_t;
  
  H5DensityReader( std::string const filename )
  : _filename(filename), _datasetname("density") {
    _file = 0;
    _is_open = true;
    try {
      _file = new H5::H5File(_filename.c_str(), H5F_ACC_RDWR);
    }
    catch(const H5::FileIException &) {
      _is_open = false;
    }
  }
  
  bool is_open() const {
    return _is_open;
  }
  
  void read( Value_t * const mem ) {
    if(!is_open())
      throw H5::FileIException();
    
    H5::DataSet dataset = _file->openDataSet(_datasetname.c_str());
    
    dataset.read(mem, H5::PredType::NATIVE_FLOAT);
  }
  
  ~H5DensityReader() {
    delete _file;
  }
  
private:
  std::string _filename, _datasetname;
  H5::H5File * _file;
  bool  _is_open;
};

#endif	/* H5DENSITYREADER_HPP */

