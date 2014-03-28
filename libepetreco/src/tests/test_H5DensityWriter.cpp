#include "H5DensityWriter.hpp"

class MyGrid
{
  public:
    
    MyGrid( float o0, float o1, float o2,
            float v0, float v1, float v2,
            int n0,   int n1,   int n2 )
    {
      _origin[0] = o0; _origin[1] = o1; _origin[2] = o2;
      _voxelSize[0] = v0; _voxelSize[1] = v1; _voxelSize[2] = v2;
      _numberOfVoxels[0] = n0; _numberOfVoxels[1] = n1; _numberOfVoxels[2] = n2;
    }

    void getOrigin( float * origin )
    {
      for(int dim=0; dim<3; dim++)
        origin[dim] = _origin[dim];
    }

    void getVoxelSize( float * voxelSize )
    {
      for(int dim=0; dim<3; dim++)
        voxelSize[dim] = _voxelSize[dim];
    }

    void getNumberOfVoxels( int * numberOfVoxels )
    {
      for(int dim=0; dim<3; dim++)
        numberOfVoxels[dim] = _numberOfVoxels[dim];
    }


  private:
    
    float _origin[3], _voxelSize[3];
    int _numberOfVoxels[3];
};



int main()
{
  MyGrid grid(.1,.2,.3,1.,2.,3.,3,4,5);
  H5DensityWriter<MyGrid> writer(std::string("test_H5DensityWriter_output.h5"));
  float * mem = new float[3*4*5];
  for(int i=0; i<3*4*5; i++)
    mem[i] = i*10.1;
  writer.write(mem, grid);

  delete mem;

  return 0;
}
