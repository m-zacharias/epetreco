#include "PlyRectangle.hpp"

#include <sstream>

#ifdef DEBUG
#include <iostream>
#endif

PlyRectangle::PlyRectangle( std::string const name,
                            Vertex const p0,
                            Vertex const p1,
                            Vertex const p2,
                            Vertex const p3 )
: PlyGeometry(name),
  _p0(p0), _p1(p1), _p2(p2), _p3(p3)
{
#ifdef DEBUG
  std::cout << "PlyRectangle::PlyRectangle(std::string const)" << std::endl;
#endif
}

PlyRectangle::~PlyRectangle()
{
#ifdef DEBUG
  std::cout << "PlyRectangle::~PlyRectangle()" << std::endl;
#endif
}

int PlyRectangle::numVertices()
{
#ifdef DEBUG
  std::cout << "PlyRectangle::numVertices()" << std::endl;
#endif
  return 4;
}

int PlyRectangle::numFaces()
{
#ifdef DEBUG
  std::cout << "PlyRectangle::numFaces()" << std::endl;
#endif
  return 1;
}

std::string PlyRectangle::verticesStr()
{
#ifdef DEBUG
  std::cout << "PlyRectangle::verticesStr()" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss << _p0.x << " " << _p0.y << " " << _p0.z << " 0 0 0 " << std::endl
     << _p1.x << " " << _p1.y << " " << _p1.z << " 0 0 0 " << std::endl
     << _p2.x << " " << _p2.y << " " << _p2.z << " 0 0 0 " << std::endl
     << _p3.x << " " << _p3.y << " " << _p3.z << " 0 0 0 " << std::endl;
  return ss.str();
}

std::string PlyRectangle::facesStr()
{
#ifdef DEBUG
  std::cout << "PlyRectangle::facesStr()" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss << "4 0 1 2 3 " << std::endl;
  return ss.str();
}

std::string PlyRectangle::facesStr( int & vertexId )
{
#ifdef DEBUG
  std::cout << "PlyRectangle::facesStr(int &)" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss << "4 " << vertexId   << " " << vertexId+1 << " "
             << vertexId+2 << " " << vertexId+3 << std::endl;
  vertexId += 4;
  return ss.str();
}


Vertex & PlyRectangle::p0()
{
  return _p0;
}

Vertex & PlyRectangle::p1()
{
  return _p1;
}

Vertex & PlyRectangle::p2()
{
  return _p2;
}

Vertex & PlyRectangle::p3()
{
  return _p3;
}

