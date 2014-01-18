#include "PlyGeometry.hpp"

#include "Iterator.hpp"
#include "ListIterator.hpp"
#include "List.hpp"

#ifdef DEBUG
#include <iostream>
#endif

PlyGeometry::~PlyGeometry()
{
#ifdef DEBUG
  std::cout << "PlyGeometry::~PlyGeometry()" << std::endl;
#endif
}

std::string PlyGeometry::name()
{
#ifdef DEBUG
  std::cout << "PlyGeometry::name()" << std::endl;
#endif
  return _name;
}


int PlyGeometry::numVertices()
{
#ifdef DEBUG
  std::cout << "PlyGeometry::numVertices()" << std::endl;
#endif
  return 0;
}

int PlyGeometry::numFaces()
{
#ifdef DEBUG
  std::cout << "PlyGeometry::numFaces()" << std::endl;
#endif
  return 0;
}

std::string PlyGeometry::verticesStr()
{
#ifdef DEBUG
  std::cout << "PlyGeometry::verticesStr()" << std::endl;
#endif
  return std::string("");
}

std::string PlyGeometry::facesStr()
{
#ifdef DEBUG
  std::cout << "PlyGeometry::facesStr()" << std::endl;
#endif
  return std::string("");
}

std::string PlyGeometry::facesStr( int & vertexId )
{
#ifdef DEBUG
  std::cout << "PlyGeometry::facesStr(int &)" << std::endl;
#endif
  return std::string("");
}


void PlyGeometry::add( PlyGeometry * g_ptr )
{
#ifdef DEBUG
  std::cout << "PlyGeometry::add(PlyGeometry *)" << std::endl;
#endif
  throw 1;
}

void PlyGeometry::remove( PlyGeometry * g_ptr )
{
#ifdef DEBUG
  std::cout << "PlyGeometry::remove(PlyGeometry *)" << std::endl;
#endif
  throw 1;
}


Iterator<PlyGeometry *> * PlyGeometry::createIterator()
{
#ifdef DEBUG
  std::cout << "PlyGeometry::createIterator()" << std::endl;
#endif
//  Iterator<PlyGeometry *> * NullIterator = 0;
//  return NullIterator;
  return 0;
}


PlyGeometry::PlyGeometry( std::string const name )
: _name(name)
{
#ifdef DEBUG
  std::cout << "PlyGeometry::PlyGeometry(std::string const)" << std::endl;
#endif
}
