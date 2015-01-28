/** @file H5DensityWriter.hpp */
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
      
      float origin[3];
      float voxelSize[3];
      int   numberOfVoxels[3];
      grid.getOrigin(        origin);
      grid.getVoxelSize(     voxelSize);
      grid.getNumberOfVoxels(numberOfVoxels);
      float max[3];
      for(int dim=0; dim<3; dim++) max[dim] =
              origin[dim] + numberOfVoxels[dim] * voxelSize[dim];
      
      /* Create the file */
      H5::H5File * file = new H5::H5File(_filename.c_str(), H5F_ACC_TRUNC);
      
// TODO: Write linearized voxel indices on grid - i.e. structured similarly to
//       the density dat itself.
//      file->createGroup("/help")
      
      /* Create dataset densdataset */
      hsize_t densdims[3] = {numberOfVoxels[0],/* Create dataspace ... */
                             numberOfVoxels[1],
                             numberOfVoxels[2]};
      H5::DataSpace densspace(3, densdims);
      H5::DataSet * densdataset;/* Create dataset ... */
      densdataset = new H5::DataSet(file->createDataSet(
                            "density", H5::PredType::NATIVE_FLOAT, densspace));
      
      /* Create attributes */
      H5::Attribute xminatt;
      xminatt = densdataset->createAttribute(
            "xmin", H5::PredType::NATIVE_FLOAT, H5S_SCALAR);
      xminatt.write(H5::PredType::NATIVE_FLOAT, &origin[0]);

      H5::Attribute yminatt;
      yminatt = densdataset->createAttribute(
            "ymin", H5::PredType::NATIVE_FLOAT, H5S_SCALAR);
      yminatt.write(H5::PredType::NATIVE_FLOAT, &origin[1]);
      
      H5::Attribute zminatt;
      zminatt = densdataset->createAttribute(
            "zmin",
            H5::PredType::NATIVE_FLOAT, H5S_SCALAR);
      zminatt.write(H5::PredType::NATIVE_FLOAT, &origin[2]);

      H5::Attribute xmaxatt;
      xmaxatt = densdataset->createAttribute(
            "xmax",
            H5::PredType::NATIVE_FLOAT, H5S_SCALAR);
      xmaxatt.write(H5::PredType::NATIVE_FLOAT, &max[0]);

      H5::Attribute ymaxatt;
      ymaxatt = densdataset->createAttribute(
            "ymax",
             H5::PredType::NATIVE_FLOAT, H5S_SCALAR);
      ymaxatt.write(H5::PredType::NATIVE_FLOAT, &max[1]);
      
      H5::Attribute zmaxatt;
      zmaxatt = densdataset->createAttribute(
            "zmax",
            H5::PredType::NATIVE_FLOAT, H5S_SCALAR);
      zmaxatt.write(H5::PredType::NATIVE_FLOAT, &max[2]);

      H5::Attribute xnbinatt;
      xnbinatt = densdataset->createAttribute(
            "xnbin",
            H5::PredType::NATIVE_INT, H5S_SCALAR);
      xnbinatt.write(H5::PredType::NATIVE_INT, &numberOfVoxels[0]);

      H5::Attribute ynbinatt;
      ynbinatt = densdataset->createAttribute(
            "ynbin",
            H5::PredType::NATIVE_INT, H5S_SCALAR);
      ynbinatt.write(H5::PredType::NATIVE_INT, &numberOfVoxels[1]);

      H5::Attribute znbinatt;
      znbinatt = densdataset->createAttribute(
            "znbin",
            H5::PredType::NATIVE_INT, H5S_SCALAR);
      znbinatt.write(H5::PredType::NATIVE_INT, &numberOfVoxels[2]);

      /* Write dataset */
      densdataset->write(mem, H5::PredType::NATIVE_FLOAT);/* Write dens data */

      /* Release resources */
      delete densdataset;
      delete file;
    }
};

#endif  // #ifndef H5DENSITYWRITER_HPP
