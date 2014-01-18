#include "CompositePlyGeometry.hpp"

#ifdef DEBUG
#include <iostream>
#endif

CompositePlyGeometry::~CompositePlyGeometry()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::~CompositePlyGeometry()" << std::endl;
#endif
}

int CompositePlyGeometry::numVertices()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::numVertices()" << std::endl;
#endif
  int total = 0;

  Iterator<PlyGeometry *> * it = CompositePlyGeometry::createIterator();
  for(it->first(); !it->isDone(); it->next()) {
    total += it->currentItem()->numVertices();
  }

  delete it;
  return total;
}

int CompositePlyGeometry::numFaces()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::numFaces()" << std::endl;
#endif
  int total = 0;

  Iterator<PlyGeometry *> * it = CompositePlyGeometry::createIterator();
  for(it->first(); !it->isDone(); it->next()) {
    total += it->currentItem()->numFaces();
  }

  delete it;
  return total;
}

std::string CompositePlyGeometry::verticesStr()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::verticesStr()" << std::endl;
#endif
  std::string str("");

  Iterator<PlyGeometry *> * it = CompositePlyGeometry::createIterator();
  for(it->first(); !it->isDone(); it->next()) {
    str += it->currentItem()->verticesStr();
  }

  delete it;
  return str;
}

std::string CompositePlyGeometry::facesStr()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::facesStr()" << std::endl;
#endif
  int vertexId = 0;
  std::string str("");
  
  Iterator<PlyGeometry *> * it = CompositePlyGeometry::createIterator();
  for(it->first(); !it->isDone(); it->next()) {
    str += it->currentItem()->facesStr(vertexId);
  }

  delete it;
  return str;
}

std::string CompositePlyGeometry::facesStr( int & vertexId )
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::facesStr(int &)" << std::endl;
#endif
  std::string str("");

  Iterator<PlyGeometry *> * it = CompositePlyGeometry::createIterator();
  for(it->first(); !it->isDone(); it->next()) {
    str += it->currentItem()->facesStr(vertexId);
  }

  delete it;
  return str;
}


void CompositePlyGeometry::add( PlyGeometry * g_ptr )
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::add()" << std::endl;
#endif
  _geometryList.append( g_ptr );
}

void CompositePlyGeometry::remove( PlyGeometry * g_ptr )
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::remove()" << std::endl;
#endif
  _geometryList.remove( g_ptr );
}

#include "ListIterator.hpp"
Iterator<PlyGeometry *> * CompositePlyGeometry::createIterator()
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::createIterator()" << std::endl;
#endif
  return new ListIterator<PlyGeometry *>(&_geometryList);
}


CompositePlyGeometry::CompositePlyGeometry( std::string const name )
: PlyGeometry(name)
{
#ifdef DEBUG
  std::cout << "CompositePlyGeometry::CompositePlyGeometry(std::string const)" << std::endl;
#endif
}
