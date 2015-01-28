/** @file test_DevelChannel.cpp */
#include "DevelChannel.hpp"
#include "DevelSetup.hpp"

#include <iostream>

class MySetup : public DevelSetup
{
  public:
    
    MySetup(int n)
    : DevelSetup(n) {}

    void setChannel( int linearChannelId, DevelChannel & channel )
    {
      int dimensionalChannelId[5];
      getDimensionalChannelId(linearChannelId, dimensionalChannelId);

      getPos0(dimensionalChannelId, channel._pos0 );
      getPos1(dimensionalChannelId, channel._pos1 );
      channel._angle=dimensionalChannelId[0]*2.;
    }
};



int main()
{
  MySetup      setup(5140980);
  DevelChannel channel;
  setup.setChannel(2000000, channel);

  DevelChannel::Coord_t pos0[3], pos1[3], edges[3];

  channel.getPos0(pos0);
  channel.getPos1(pos1);
  channel.getEdges(edges);
  int angle = channel.getAngle();

  std::cout << "Channel:" << std::endl;
  std::cout << "pos0: " << pos0[0] << ", " << pos0[1] << ", " << pos0[2] << std::endl
            << "pos1: " << pos1[0] << ", " << pos1[1] << ", " << pos1[2] << std::endl
            << "edges: " << edges[0] << ", " << edges[1] << ", " << edges[2] << std::endl
            << "angle: " << angle << std::endl;

  DevelChannel::Ray_t * ray = new DevelChannel::Ray_t;
  channel.createRays(ray, 1);
  DevelChannel::Ray_t::Vertex_t start(ray->start());
  DevelChannel::Ray_t::Vertex_t end(ray->end());

  std::cout << "Ray:" << std::endl;
  std::cout << "start: " << start[0] << ", " << start[1] << ", " << start[2] << std::endl
            << "end: " << end[0] << ", " << end[1] << ", " << end[2] << std::endl;

  delete ray;
  return 0;
}
