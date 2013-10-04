#include "PlyWriter.hpp"
#include <typeinfo>

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
  
  cout << "Create Box ..." << flush;
  Box box = Box( p0, p1, p2, p3, p4, p5, p6, p7 );
  cout << " done" << endl;
  
  cout << "Create Scene ..." << flush;
  Scene scene;
  scene.add_Geometry( &box );
  cout << " done" << endl;
  
  cout << "Create Writer ..." << flush;
  PlyWriter writer( "test.ply" );
  cout << " done" << endl << "Write ..." << flush;
  writer.write( scene );
  cout << " done" << endl << "Close ..." << flush;
  writer.close();
  cout << " done" << endl;
  
  return 0;
}
