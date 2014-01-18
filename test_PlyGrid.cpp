#include "PlyGrid.hpp"
#include "PlyWriter.hpp"

int main()
{
  PlyGrid grid("grid",
               Vertex(0.,0.,0.),
               3, 4, 5,
               1., 1., 1.
              );

  PlyWriter writer("test_PlyGrid_output.ply");
  writer.write(grid);
  writer.close();

  return 0;
}
