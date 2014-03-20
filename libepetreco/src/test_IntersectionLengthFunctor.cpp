#include "IntersectionLengthFunctor.hpp"
#include "TestRay.hpp"
#include "TestGrid.hpp"
#include "Ply.hpp"

#include <iostream>

template<typename CoordType>
struct Intersection
{
  typedef CoordType Coord_t;

  void setId( int * id )
  {
    for(int dim=0; dim<3; dim++)
      _id[dim] = id[dim];
  }

  void setLength( Coord_t length )
  {
    _length = length;
  }

  int _id[3];
  Coord_t _length;
};
typedef Intersection<double> TestIntersection;


typedef IntersectionLengthFunctor<TestRay, TestGrid, TestIntersection> TestFunctor;


int main()
{
  TestGrid grid(TestGrid::Vertex_t(0., 0., 0.), TestGrid::Vertex_t(1., 1.,1.), 2, 2, 1);
  TestRay  ray (TestRay::Vertex_t (-0.5,-0.5,0.),  TestRay::Vertex_t (3.5,2.5,1.));

  TestFunctor functor;
  int nisc = functor.getNCrossedVoxels(ray,grid);
  std::cout << "Number of voxels crossed: " << nisc << std::endl;

  TestIntersection * iscs = new TestIntersection[nisc];
  functor.calculateIntersectionLengths(iscs, ray, grid);

  for(int i=0; i<nisc; i++)
  {
    std::cout << "id: ("
              << iscs[i]._id[0] << "," << iscs[i]._id[1] << "," << iscs[i]._id[2]
              << "), intersection length: "
              << iscs[i]._length << std::endl;
  }

  PlyWriter gridwriter("test_IntersectionLengthFunctor_grid.ply");
  gridwriter.write(grid);
  gridwriter.close();

  PlyWriter raywriter("test_IntersectionLengthFunctor_ray.ply");
  raywriter.write(ray);
  raywriter.close();

  return 0;
}
