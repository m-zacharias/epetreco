/* 2014-03-18
 * Example file for writing HDF5 files, including a group.
 */
#include "H5Cpp.h"
#include <string>
#include <iostream>
#include <cstdlib>

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

int main()
{
  std::string const fn("example1_H5write_output.h5");
  std::string const gn("/the_group");
  int const         DIM1 = 3;
  int const         DIM2 = 7;
  
  /* Create file */
  H5File *          file = new H5File(fn.c_str(), H5F_ACC_TRUNC);
  
  /* Create group */
  file->createGroup(gn.c_str());

  /* Create dataspace for the dataset in the file */
  hsize_t           fdims[] = {DIM1, DIM2};
  DataSpace         fspace(2, fdims);

  /* Create property list for the dataset and set up fillvalue */
  int               fillval = 0;
  DSetCreatPropList plist;
  plist.setFillValue(PredType::NATIVE_INT, &fillval);

  /* Create dataset and write it into the file */
  std::string       datasetname = gn+std::string("/zeros_in_file");
  DataSet *         dataset = new DataSet(file->createDataSet(
                                        datasetname, PredType::NATIVE_INT,
                                        fspace, plist));

  delete dataset;
  delete file;

  exit(EXIT_SUCCESS);
}
