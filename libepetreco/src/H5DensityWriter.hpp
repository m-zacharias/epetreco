#ifndef H5DENSITYWRITER_HPP
#define H5DENSITYWRITER_HPP

#include "H5Cpp.h"
#include <string>
#include <iostream>

/**
 * @brief Class template for writing density data to an HDF5 file
 * 
 * The grid type TGrid has to implement low level getter functions that write
 * array information of 3 elements to the arrays given by their arguments:
 *  - for the grid origin:      void getOrigin(flaot*)
 *  - for the grid voxelsize:   void getVoxelSize(float*)
 *  - for the grid dimensions:  void getNumberOfVoxels(int*)
 */
template<typename TGrid>
class H5DensityWriter
{
  private:
    
    std::string _filename;


  public:
    
    /**
     * @brief Constructor
     *
     * @param filename Name of the HDF5 file to write into. Former content will
     * be truncated.
     */
    H5DensityWriter( std::string const & filename )
    : _filename(filename)
    {
#ifdef DEBUG
      std::cout << "H5DensityWriter::H5DensityWriter(std::string const &)"
                << std::endl;
#endif
    }
    
    /**
     * @brief Write data into file
     *
     * @param mem Pointer to raw density data
     * @param grid Density grid object that provides origin, voxelsize and
     * dimensional voxel number information
     */
    void write( float * const mem, TGrid & grid )
    {
#ifdef DEBUG
      std::cout << "H5DensityWriter::write(Value_t * const mem, TGrid grid)"
                << std::endl;
#endif
      /* Create the file */
      H5::H5File * file = new H5::H5File(_filename.c_str(), H5F_ACC_TRUNC);
      
      ///* Create and write dataset density_grid */
      //struct Grid_t             /* Define struct ... */
      //{
      //  float origin[3], voxelSize[3];
      //  int numberOfVoxels[3];
      //};
      //hsize_t dims[1] = {3};    /* Create datatype ... */
      //H5::ArrayType vtype( H5::PredType::NATIVE_FLOAT, 1, dims);
      //H5::ArrayType i3type(H5::PredType::NATIVE_INT,   1, dims);
      //H5::CompType  gtype(sizeof(Grid_t));
      //gtype.insertMember(
      //            "origin",           HOFFSET(Grid_t, origin),         vtype);
      //gtype.insertMember(
      //            "voxelsize",        HOFFSET(Grid_t, voxelSize),      vtype);
      //gtype.insertMember(
      //            "number_of_voxels", HOFFSET(Grid_t, numberOfVoxels), i3type);
      //hsize_t ngrids[1] = {1};   /* Create the dataspace ... */
      //H5::DataSpace gridspace(1, ngrids);
      //H5::DataSet * griddataset;/* Create the dataset ... */
      //griddataset = new H5::DataSet(file->createDataSet(
      //                      "density_grid", gtype, gridspace));
      //Grid_t fgrid;             /* Get grid data ... */
      //grid.getOrigin(        fgrid.origin);
      //grid.getVoxelSize(     fgrid.voxelSize);
      //grid.getNumberOfVoxels(fgrid.numberOfVoxels);
      //griddataset->write(&fgrid, gtype);/* Write grid data */
      float origin[3];
      float voxelSize[3];
      int   numberOfVoxels[3];
      grid.getOrigin(        origin);
      grid.getVoxelSize(     voxelSize);
      grid.getNumberOfVoxels(numberOfVoxels);
      float max[3];
      for(int dim=0; dim<3; dim++) max[dim] =
              origin[dim] + numberOfVoxels[dim] * voxelSize[dim];


      H5::Group * griddefinition = new H5::Group(
            file->createGroup("/grid definition"));


      hsize_t helper_one[1] = {1};
      H5::DataSpace singlespace(1, helper_one);
      
      
      H5::DataSet * xmindataset;
      xmindataset = new H5::DataSet(
            file->createDataSet(
                  "/grid definition/xmin",
                  H5::PredType::NATIVE_FLOAT, singlespace));
      xmindataset->write(&origin[0], H5::PredType::NATIVE_FLOAT);

      H5::DataSet * ymindataset;
      ymindataset = new H5::DataSet(
            file->createDataSet(
                  "/grid definition/ymin",
                  H5::PredType::NATIVE_FLOAT, singlespace));
      ymindataset->write(&origin[1], H5::PredType::NATIVE_FLOAT);
      
      H5::DataSet * zmindataset;
      zmindataset = new H5::DataSet(
            file->createDataSet(
                  "/grid definition/zmin",
                  H5::PredType::NATIVE_FLOAT, singlespace));
      zmindataset->write(&origin[2], H5::PredType::NATIVE_FLOAT);

      H5::DataSet * xmaxdataset;
      xmaxdataset = new H5::DataSet(
            file->createDataSet(
                  "/grid definition/xmax",
                  H5::PredType::NATIVE_FLOAT, singlespace));
      xmaxdataset->write(&max[0], H5::PredType::NATIVE_FLOAT);

      H5::DataSet * ymaxdataset;
      ymaxdataset = new H5::DataSet(
            file->createDataSet(
                  "/grid definition/ymax",
                  H5::PredType::NATIVE_FLOAT, singlespace));
      ymaxdataset->write(&max[1], H5::PredType::NATIVE_FLOAT);
      
      H5::DataSet * zmaxdataset;
      zmaxdataset = new H5::DataSet(
            file->createDataSet(
                  "/grid definition/zmax",
                  H5::PredType::NATIVE_FLOAT, singlespace));
      zmaxdataset->write(&max[2], H5::PredType::NATIVE_FLOAT);

      H5::DataSet * xnbindataset;
      xnbindataset = new H5::DataSet(
            file->createDataSet(
                  "/grid definition/xnbin",
                  H5::PredType::NATIVE_INT, singlespace));
      xnbindataset->write(&numberOfVoxels[0], H5::PredType::NATIVE_INT);

      H5::DataSet * ynbindataset;
      ynbindataset = new H5::DataSet(
            file->createDataSet(
                  "/grid definition/ynbin",
                  H5::PredType::NATIVE_INT, singlespace));
      ynbindataset->write(&numberOfVoxels[1], H5::PredType::NATIVE_INT);

      H5::DataSet * znbindataset;
      znbindataset = new H5::DataSet(
            file->createDataSet(
                  "/grid definition/znbin",
                  H5::PredType::NATIVE_INT, singlespace));
      znbindataset->write(&numberOfVoxels[2], H5::PredType::NATIVE_INT);

// TODO: Write linearized voxel indices on grid - i.e. structured similarly to
//       the density dat itself.
//      file->createGroup("/help")
      
      /* Create and write dataset density */
      hsize_t densdims[3] = {numberOfVoxels[0],/* Create dataspace ... */
                             numberOfVoxels[1],
                             numberOfVoxels[2]};
      H5::DataSpace densspace(3, densdims);
      H5::DataSet * densdataset;/* Create dataset ... */
      densdataset = new H5::DataSet(file->createDataSet(
                            "density", H5::PredType::NATIVE_FLOAT, densspace));
      densdataset->write(mem, H5::PredType::NATIVE_FLOAT);/* Write dens data */

      /* Release resources */
      delete densdataset;
      delete xmindataset;
      delete ymindataset;
      delete zmindataset;
      delete xmaxdataset;
      delete ymaxdataset;
      delete zmaxdataset;
      delete xnbindataset;
      delete ynbindataset;
      delete znbindataset;
      delete file;
    }
};

#endif  // #ifndef H5DENSITYWRITER_HPP
