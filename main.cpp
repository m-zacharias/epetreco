#include "PlyWriter.hpp"

using namespace std;

int main( void ) {
  coord_type p0[3] = { 0., 0., 0. };
  coord_type p1[3] = { 1., 0., 0. };
  coord_type p2[3] = { 1., 1., 0. };
  coord_type p3[3] = { 0., 1., 0. };
  coord_type p4[3] = { 0., 0., 1. };
  coord_type p5[3] = { 1., 0., 1. };
  coord_type p6[3] = { 1., 1., 1. };
  coord_type p7[3] = { 0., 1., 1. };
  
  //~ Rectangle rect = Rectangle( p0, p1, p2, p3 );
  //~ Triangle tri = Triangle( p0, p1, p2 );
  Box box = Box( p0, p1, p2, p3, p4, p5, p6, p7 );
  
  Scene scene;
  //~ scene.add_Geometry( &tri );
  //~ scene.add_Geometry( &rect );
  scene.add_Geometry( &box );
  
  PlyWriter writer( "test.ply" );
  writer.write( scene );
  writer.close();
  
  return 0;
}
