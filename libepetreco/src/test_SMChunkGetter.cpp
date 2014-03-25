#include "SMChunkGetter.hpp"
#include "DevelChannel.hpp"
#include "DevelSetup.hpp"
#include "DevelSMChunk.hpp"
#include "Ply.hpp"

int main()
{
  std::cout << "DevelSetup setup(5140980);" << std::endl;
  DevelSetup setup(5140980);

  std::cout
  << "DevelGrid  grid(DevelGrid::Vertex_t(0,0,0), DevelGrid::Vertex_t(0.1,0.1,0.1), 1,1,1);"
  << std::endl;
  DevelGrid  grid(DevelGrid::Vertex_t(0,0,0), DevelGrid::Vertex_t(0.1,0.1,0.1), 1,1,1);
  
  std::cout
  << "DevelSMChunk sm(setup, grid);"
  << std::endl;
  DevelSMChunk sm(setup, grid);

  std::cout
  << "SMChunkGetter<DevelSetup, DevelGrid, DevelSMChunk>()(setup, grid, 0, 10000, sm, 1);"
  << std::endl;
  SMChunkGetter<DevelSetup, DevelGrid, DevelSMChunk>()(setup, grid, 0, 10000, sm, 1);
  
  return 0;
}
