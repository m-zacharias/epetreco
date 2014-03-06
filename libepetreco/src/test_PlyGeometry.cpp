#include "PlyGeometry.hpp"
#include "CompositePlyGeometry.hpp"
#include "PlyRectangle.hpp"

#include <iostream>
#include <string>

class Scene : public CompositePlyGeometry
{
  public:
    
    Scene( std::string const name )
    : CompositePlyGeometry(name) {}

    int count()
    {
      return this->_geometryList.count();
    }
};

int main( void )
{
  PlyRectangle rect( std::string("rect"),
                     Vertex(0.,0.,0.),
                     Vertex(0.,1.,0.),
                     Vertex(1.,1.,0.),
                     Vertex(1.,0.,0.)
                   );
  
  Scene scene( std::string("scene") );

  scene.add(&rect);
  scene.add(&rect);
  scene.add(&rect);
  std::cout << "count: " << scene.count() << std::endl;

  Iterator<PlyGeometry*>* it = scene.createIterator();
  std::cout << "starting iterations..." << std::endl;
  for(it->first(); !it->isDone(); it->next()) {
    std::cout << it->currentItem()->name() << std::endl << std::flush;
  }

  std::cout << scene.verticesStr() << scene.facesStr() << std::endl;

  return 0;
}
