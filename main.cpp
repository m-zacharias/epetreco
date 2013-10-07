#include "PlyWriter.hpp"
#include <typeinfo>

using namespace std;

int main( void ) {
  Vertex p0( 0., 0., 0. );
  Vertex p1( 1., 0., 0. );
  Vertex p2( 1., 1., 0. );
  Vertex p3( 0., 1., 0. );
  Vertex p4( 0., 0., 1. );
  Vertex p5( 1., 0., 1. );
  Vertex p6( 1., 1., 1. );
  Vertex p7( 0., 1., 1. );
  
  #ifdef DEBUG
  cout << "Create Box ..." << flush;
  #endif
  Box box = Box( p0, p1, p2, p3, p4, p5, p6, p7 );
  #ifdef DEBUG
  cout << " done" << endl;
  #endif
  //~ #ifdef DEBUG
  //~ cout << "Create multiple objects Line ..." << flush;
  //~ #endif
  //~ Line line0 = Line( p0, p1 );
  //~ Line line1 = Line( p0, p3 );
  //~ Line line2 = Line( p0, p4 );
  //~ #ifdef DEBUG
  //~ cout << " done" << endl;
  //~ #endif
  
  #ifdef DEBUG
  cout << "Create Scene ..." << flush;
  #endif
  Scene scene;
  scene.add_Geometry( &box );
  //~ scene.add_Geometry( &line0 );
  //~ scene.add_Geometry( &line1 );
  //~ scene.add_Geometry( &line2 );
  #ifdef DEBUG
  cout << " done" << endl;
  #endif
  
  #ifdef DEBUG
  cout << "Create Writer ..." << flush;
  #endif
  PlyWriter writer( "test.ply" );
  #ifdef DEBUG
  cout << " done" << endl << "Write ..." << flush;
  #endif
  writer.write( scene );
  #ifdef DEBUG
  cout << " done" << endl << "Close ..." << flush;
  #endif
  writer.close();
  #ifdef DEBUG
  cout << " done" << endl;
  #endif
  
  return 0;
}
