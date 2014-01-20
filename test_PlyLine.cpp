#include "PlyLine.hpp"
#include "PlyWriter.hpp"

int main()
{
  PlyLine line("line",
               Vertex(0.,0.,0.),
               Vertex(1.,1.,1.)
          );
  PlyWriter writer("test_PlyLine_output.ply");
  writer.write(line);
  writer.close();

  return 0;
}
