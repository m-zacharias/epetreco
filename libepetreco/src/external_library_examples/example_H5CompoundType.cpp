/* 2014-03-18
 * Example: Create a compound datatype, write data of this type into a file.
 * 
 * Follows this example:
 * http://www.hdfgroup.org/HDF5/doc/cpplus_RM/compound_8cpp-example.html
 */
#include "H5Cpp.h"

#ifndef H5_NO_NAMESPACE
using namespace H5;
#endif

int main()
{
  std::string const FILENAME("example_H5CompoundType_output.h5");
  std::string const DATASETNAME("grid_definitions");
  int const         RANK = 1;
  int const         LENGTH = 5;

  /* Define struct */
  struct Grid
  {
    float origin[3], voxelsize[3];
    int   number_of_voxels[3];
  };

  /* Create the memory datatype */
  hsize_t dims[1] = {3};
  ArrayType vertextype(PredType::NATIVE_FLOAT, 1, dims);
  ArrayType int3type(  PredType::NATIVE_INT,   1, dims);
  CompType mtype(sizeof(Grid));
  mtype.insertMember("origin",           HOFFSET(Grid, origin), vertextype);
  mtype.insertMember("voxelsize",        HOFFSET(Grid, voxelsize), vertextype);
  mtype.insertMember("number_of_voxels", HOFFSET(Grid, number_of_voxels), int3type);

  /* Create the file */
  H5File * file = new H5File(FILENAME, H5F_ACC_TRUNC);

  /* Create the dataspace */
  hsize_t dim[] = {LENGTH};
  DataSpace dataspace(RANK, dim);

  /* Create the dataset */
  DataSet * dataset = new DataSet(file->createDataSet(
                                              DATASETNAME, mtype, dataspace));
  /* Create data */
  Grid grids[LENGTH];
  for(int i=0; i<LENGTH; i++)
  {
    grids[i].origin[0] = .1*i;
    grids[i].origin[1] = i;
    grids[i].origin[2] = 10.*i;
    grids[i].voxelsize[0] = 1.-.1*i;
    grids[i].voxelsize[1] = 10.-i;
    grids[i].voxelsize[2] = 100.-10.*i;
    grids[i].number_of_voxels[0] = i;
    grids[i].number_of_voxels[1] = i*i;
    grids[i].number_of_voxels[2] = i*i*i;
  }
  
  /* Write data to the dataset */
  dataset->write(grids, mtype);
  
  /* Release the resources */
  delete dataset;
  delete file;

  return 0;
}
