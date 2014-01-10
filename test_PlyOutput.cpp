#include <iostream>
#include "PlyOutput.hpp"

using namespace PlyOutput;

int main( void )
{
  /* Create a box */
  Box box( Vertex( 0., 0., 0. ), 1., 1., 1. );

  /* Create a line */
  Vertex l0( -1., -1., -1. );
  Vertex l1(  1.,  1.5, 1.5 );
  Line line( l0, l1 );

  /* Create a scene, register box and line */
  Scene scene;
  scene.add_Geometry( &box );
  scene.add_Geometry( &line );

  /* Create a writer, write scene, close writer */
  PlyWriter writer( "test_PlyOutput_output.ply" );
  writer.write( scene );
  writer.close();

  return 0;
}
