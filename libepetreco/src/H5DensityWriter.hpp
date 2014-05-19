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
      
      /* Create and write dataset density_grid */
      struct Grid_t             /* Define struct ... */
      {
        float origin[3], voxelSize[3];
        int numberOfVoxels[3];
      };
      hsize_t dims[1] = {3};    /* Create datatype ... */
      H5::ArrayType vtype( H5::PredType::NATIVE_FLOAT, 1, dims);
      H5::ArrayType i3type(H5::PredType::NATIVE_INT,   1, dims);
      H5::CompType  gtype(sizeof(Grid_t));
      gtype.insertMember(
                  "origin",           HOFFSET(Grid_t, origin),         vtype);
      gtype.insertMember(
                  "voxelsize",        HOFFSET(Grid_t, voxelSize),      vtype);
      gtype.insertMember(
                  "number_of_voxels", HOFFSET(Grid_t, numberOfVoxels), i3type);
      hsize_t ngrids[1] = {1};   /* Create the dataspace ... */
      H5::DataSpace gridspace(1, ngrids);
      H5::DataSet * griddataset;/* Create the dataset ... */
      griddataset = new H5::DataSet(file->createDataSet(
                            "density_grid", gtype, gridspace));
      Grid_t fgrid;             /* Get grid data ... */
      grid.getOrigin(        fgrid.origin);
      grid.getVoxelSize(     fgrid.voxelSize);
      grid.getNumberOfVoxels(fgrid.numberOfVoxels);
      griddataset->write(&fgrid, gtype);/* Write grid data */

// TODO: Write linearized voxel indices on grid - i.e. structured similarly to
//       the density dat itself.
//      file->createGroup("/help")
      
      /* Create and write dataset density */
      hsize_t densdims[3] = {fgrid.numberOfVoxels[0],/* Create dataspace ... */
                             fgrid.numberOfVoxels[1],
                             fgrid.numberOfVoxels[2]};
      H5::DataSpace densspace(3, densdims);
      H5::DataSet * densdataset;/* Create dataset ... */
      densdataset = new H5::DataSet(file->createDataSet(
                            "density", H5::PredType::NATIVE_FLOAT, densspace));
      densdataset->write(mem, H5::PredType::NATIVE_FLOAT);/* Write dens data */

      /* Release resources */
      delete densdataset;
      delete griddataset;
      delete file;
    }
};

#endif  // #ifndef H5DENSITYWRITER_HPP
