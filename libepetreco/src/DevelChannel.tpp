#include "DevelChannel.hpp"
#include <cstdlib> // for random numbers

#define PI 3.1415927


void      DevelChannel
::_trafo(
      Coord_t * const mem_trafo, Coord_t * const edges,
      Coord_t * const pos, Coord_t const sin, Coord_t const cos )
{
#ifdef DEBUG
#ifndef NO_DEVELCHANNEL_DEBUG
  std::cout << "DevelChannel::_trafo(Coord_t * const, Coord_t * const, Coord_t"
            << " * const, Coord_t, Coord_t)" << std::endl;
#endif
#endif
  mem_trafo[0*4 + 0] = cos*edges[0];
  mem_trafo[0*4 + 1] = 0.;
  mem_trafo[0*4 + 2] = sin*edges[2];
  mem_trafo[0*4 + 3] = cos*(pos[0]-.5*edges[0])\
                      +sin*(pos[2]-.5*edges[2]);
  mem_trafo[1*4 + 0] = 0.;
  mem_trafo[1*4 + 1] = edges[1];
  mem_trafo[1*4 + 2] = 0.;
  mem_trafo[1*4 + 3] = pos[1]-.5*edges[1];
  mem_trafo[2*4 + 0] =-sin*edges[0];
  mem_trafo[2*4 + 1] = 0.;
  mem_trafo[2*4 + 2] = cos*edges[2];
  mem_trafo[2*4 + 3] =-sin*(pos[0]-.5*edges[0])\
                      +cos*(pos[2]-.5*edges[2]);
}


void      DevelChannel
::getPos0( Coord_t * pos0 )
{
#ifdef DEBUG
#ifndef NO_DEVELCHANNEL_DEBUG
  std::cout << "DevelChannel::getPos0(Coord_t*)" << std::endl;
#endif
#endif
  pos0[0] = _pos0[0]; // x [cm]
  pos0[1] = _pos0[1]; // y [cm]
  pos0[2] = _pos0[2]; // z [cm]
}
    

void      DevelChannel
::getPos1( Coord_t * pos1 )
{
#ifdef DEBUG
#ifndef NO_DEVELCHANNEL_DEBUG
  std::cout << "DevelChannel::getPos1(Coord_t*)" << std::endl;
#endif
#endif
  pos1[0] = _pos1[0];
  pos1[1] = _pos1[1];
  pos1[2] = _pos1[2];
}


int      DevelChannel
::getAngle()
{
#ifdef DEBUG
#ifndef NO_DEVELCHANNEL_DEBUG
  std::cout << "DevelChannel::getAngle()" << std::endl;
#endif
#endif
  return _angle;
}

    
void      DevelChannel
::getEdges( Coord_t * mem_edges )
{
#ifdef DEBUG
#ifndef NO_DEVELCHANNEL_DEBUG
  std::cout << "DevelChannel::getEdges(Coord_t*)" << std::endl;
#endif
#endif
  mem_edges[0] = 2.0;   // x [cm]
  mem_edges[1] = 0.4;   // y [cm]
  mem_edges[2] = 0.4;   // z [cm]
}


void    DevelChannel
::createRays( DevelChannel::Ray_t * rays, int nRays )
{
#ifdef DEBUG
#ifndef NO_DEVELCHANNEL_DEBUG
  std::cout << "DevelChannel::createRays(Ray_t *)" << std::endl;
#endif
#endif
  // Get positions, edges, angle of det0, det1 segments
  Coord_t pos0[3];
  Coord_t pos1[3];
  Coord_t edges[3];
  getPos0(pos0);
  getPos1(pos1);
  getEdges(edges);
  Coord_t angle = (Coord_t)(getAngle())/180.*PI;
  Coord_t cos = std::cos(angle);
  Coord_t sin = std::sin(angle);

  // Generate transformation matrices for det0, det1
  Coord_t * trafo0 = new Coord_t[3*4];
  Coord_t * trafo1 = new Coord_t[3*4];
  _trafo(trafo0, edges, pos0, sin, cos);
  _trafo(trafo1, edges, pos1, sin, cos);

  // Iterate over rays
  for(int ray_id=0; ray_id<nRays; ray_id++)
  {
    // Generate cartesian randoms in homogeneous coordinates
    Coord_t s[4];
    Coord_t e[4];
    for(int i=0; i<3; i++)
    {
      s[i] = (Coord_t)(rand())/RAND_MAX;
      e[i] = (Coord_t)(rand())/RAND_MAX;
    }
    s[3] = 1.;
    e[3] = 1.;
    
    // Transform to obtain start, end
    Coord_t start[3];
    Coord_t end[3];
    for(int i=0; i<3; i++)
    {
      start[i] = 0.;
      end[i]   = 0.;

      for(int j=0; j<4; j++)
      {
        start[i] += trafo0[i*4 + j] * s[j];
        end[i]   += trafo1[i*4 + j] * e[j];
      }
    }

    // Write ray
    rays[ray_id] = DevelRay(DevelRay::Vertex_t(start[0],start[1],start[2]),
                            DevelRay::Vertex_t(end[0],  end[1],  end[2]));
  }
  delete[] trafo0;
  delete[] trafo1;
}
 

//void      DevelChannel
//::setRays( int nrays )
//{
//#ifdef DEBUG
//#ifndef NO_DEVELCHANNEL_DEBUG
//  std::cout << "DevelChannel::setRays(int)" << std::endl;
//#endif
//#endif
//  if(!_updateRayMemSize(nrays))
//    return;
//
//  // Get positions, edges, angle of det0, det1 segments
//  Coord_t pos0[3];
//  Coord_t pos1[3];
//  Coord_t edges[3];
//  getPos0(pos0);
//  getPos1(pos1);
//  getEdges(edges);
//  Coord_t angle = (Coord_t)(getAngle())/180.*PI;
//  Coord_t cos = std::cos(angle);
//  Coord_t sin = std::sin(angle);
//
//  // Generate transformation matrices for det0, det1
//  Coord_t * trafo0 = new Coord_t[3*4];
//  Coord_t * trafo1 = new Coord_t[3*4];
//  _trafo(trafo0, edges, pos0, sin, cos);
//  _trafo(trafo1, edges, pos1, sin, cos);
//
//  // Iterate over rays
//  for(int ray_id=0; ray_id<nrays; ray_id++)
//  {
//    // Generate cartesian randoms in homogeneous coordinates
//    Coord_t s[4];
//    Coord_t e[4];
//    for(int i=0; i<3; i++)
//    {
//      s[i] = (Coord_t)(rand())/RAND_MAX;
//      e[i] = (Coord_t)(rand())/RAND_MAX;
//    }
//    s[3] = 1.;
//    e[3] = 1.;
//    
//    // Transform to obtain start, end
//    Coord_t start[3];
//    Coord_t end[3];
//    for(int i=0; i<3; i++)
//    {
//      start[i] = 0.;
//      end[i]   = 0.;
//
//      for(int j=0; j<4; j++)
//      {
//        start[i] += trafo0[i*4 + j] * s[j];
//        end[i]   += trafo1[i*4 + j] * e[j];
//      }
//    }
//
//    // Write ray
//    _rays[ray_id] = DevelRay(DevelRay::Vertex_t(start[0],start[1],start[2]),
//                            DevelRay::Vertex_t(end[0],  end[1],  end[2]));
//  }
//  delete[] trafo0;
//  delete[] trafo1;
//}


//DevelChannel::PlyRepr      DevelChannel
//::getPlyRepr()
//{
//#ifdef DEBUG
//#ifndef NO_DEVELCHANNEL_DEBUG
//  std::cout << "DevelChannel::getPlyRepr" << std::endl;
//#endif
//#endif
//  PlyRepr g;
//
//// TODO: Add boxes representing the two detector segments.  Problem: There
////       is no way to rotate PlyBoxes at the moment.      
////      // Get positions, edges, angle of det0, det1 segments
////      Coord_t pos0[3];
////      Coord_t pos1[3];
////      Coord_t edges[3];
////      getPos0(pos0);
////      getPos1(pos1);
////      getEdges(edges);
////      Coord_t angle = (Coord_t)(getAngle())/180.*PI;
////      Coord_t cos = std::cos(angle);
////      Coord_t sin = std::sin(angle);
////
////      // Get transformation matrices
////      Coord_t trafo0[3][4];
////      Coord_t trafo1[3][4];
////      _trafo(trafo0, edges, pos0, sin, cos);
////      _trafo(trafo1, edges, pos1, sin, cos);
//
//  for(int i=0; i<_nrays; i++)
//  {
//    g.add(&_rays[i]);
//  }
//
//  return g;
//}


     DevelChannel
::DevelChannel( int angle, Coord_t * pos0, Coord_t * pos1 )
: _angle(angle)
{
#ifdef DEBUG
#ifndef NO_DEVELCHANNEL_DEBUG
  std::cout << "DevelChannel::DevelChannel(int,Coord_t*,Coord_t*)" << std::endl;
#endif
#endif
  for(int i=0; i<3; i++)
  {
    _pos0[i] = pos0[i];
    _pos1[i] = pos1[i];
  }
}


     DevelChannel
::DevelChannel( void )
{
#ifdef DEBUG
#ifndef NO_DEVELCHANNEL_DEBUG
  std::cout << "DevelChannel::DevelChannel()" << std::endl;
#endif
#endif
}


     DevelChannel
::DevelChannel( DevelChannel const & ori )
: _angle(ori._angle)
{
#ifdef DEBUG
#ifndef NO_DEVELCHANNEL_DEBUG
  std::cout << "DevelChannel::DevelChannel(DevelChannel const &)" << std::endl;
#endif
#endif
  for(int dim=0; dim<3; dim++)
  {
    _pos0[dim] = ori._pos0[dim];
    _pos1[dim] = ori._pos1[dim];
  }
}


void      DevelChannel
::operator=( DevelChannel const & ori )
{
#ifdef DEBUG
#ifndef NO_DEVELCHANNEL_DEBUG
  std::cout << "DevelChannel::operator=(DevelChannel const &)" << std::endl;
#endif
#endif
  _angle = ori._angle;
  
  for(int dim=0; dim<3; dim++)
  {
    _pos0[dim] = ori._pos0[dim];
    _pos1[dim] = ori._pos1[dim];
  }
}
