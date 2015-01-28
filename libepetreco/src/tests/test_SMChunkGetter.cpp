/** @file test_SMChunkGetter.cpp */
#include "SMChunkGetter.hpp"
#include "DevelChannel.hpp"
#include "DevelSetup.hpp"
#include "DevelSMChunk.hpp"
#include "Ply.hpp"
#include "FileTalk.hpp"

int main()
{
  SAYLINE(__LINE__+1);
  DevelSetup setup(5140980);

  SAYLINE(__LINE__+1);SAYLINE(__LINE__+2);
  DevelGrid  grid(DevelGrid::Vertex_t(0,0,0),
                  DevelGrid::Vertex_t(0.1,0.1,0.1), 1,1,1);
  
  SAYLINE(__LINE__+1);
  DevelSMChunk sm(setup, grid);

  SAYLINE(__LINE__+1);SAYLINE(__LINE__+2);
  SMChunkGetter<DevelSetup, DevelGrid, DevelSMChunk>()
        (setup, grid, 0, 10000, sm, 1);
  
  return 0;
}
