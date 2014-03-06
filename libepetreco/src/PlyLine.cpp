#include "PlyLine.hpp"

#include <sstream>

#ifdef DEBUG
#include <iostream>
#endif

PlyLine::PlyLine( std::string const name, Vertex const p0, Vertex const p1 )
: PlyGeometry(name), _p0(p0), _p1(p1)
{
#ifdef DEBUG
  std::cout << "PlyLine::PLyLine(std::string const, Vertex const, Vertex const)"
            << std::endl;
#endif
}

PlyLine::~PlyLine()
{
#ifdef DEBUG
  std::cout << "PlyLine::~PLyLine()" << std::endl;
#endif
}

int PlyLine::numVertices()
{
#ifdef DEBUG
  std::cout << "PlyLine::numVertices()" << std::endl;
#endif
  return 3;
}

int PlyLine::numFaces()
{
#ifdef DEBUG
  std::cout << "PlyLine::numFaces()" << std::endl;
#endif
  return 1;
}

std::string PlyLine::verticesStr()
{
#ifdef DEBUG
  std::cout << "PlyLine::verticesStr()" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss\
    << _p0.x << " " << _p0.y << " " << _p0.z << " 0 0 0 " << std::endl
    << 0.5*(_p0.x+_p1.x) << " "
    << 0.5*(_p0.y+_p1.y) << " "
    << 0.5*(_p0.z+_p1.z)                     << " 0 0 0 " << std::endl
    << _p1.x << " " << _p1.y << " " << _p1.z << " 0 0 0 " << std::endl;
  return ss.str();
}

std::string PlyLine::facesStr()
{
#ifdef DEBUG
  std::cout << "PlyLine::facesStr()" << std::endl;
#endif
  return std::string("3 0 1 2");
}

std::string PlyLine::facesStr( int & v )
{
#ifdef DEBUG
  std::cout << "PlyLine::facesStr(int &)" << std::endl;
#endif
  std::stringstream ss;
  ss.str() = "";
  ss\
    << "3 " << v << " " << v+1 << " " << v+2 << std::endl;
  v += 3;
  return ss.str();
}

Vertex & PlyLine::p0()
{
#ifdef DEBUG
  std::cout << "PlyLine::p0()" << std::endl;
#endif
  return _p0;
}

Vertex & PlyLine::p1()
{
#ifdef DEBUG
  std::cout << "PlyLine::p1()" << std::endl;
#endif
  return _p1;
}
