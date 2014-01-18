#include "PlyRectangle.hpp"
#include "CompositePlyGeometry.hpp"
#include "PlyWriter.hpp"
#include "Iterator.hpp"

#include <iostream>

class Scene : public CompositePlyGeometry
{
  public:
    
    Scene( std::string const name )
    : CompositePlyGeometry(name) {}
};

int main()
{
  PlyRectangle rect1( std::string("rect1"),
                      Vertex(0.,0.,0.),
                      Vertex(1.,0.,0.),
                      Vertex(1.,1.,0.),
                      Vertex(0.,1.,0.)
                    );
  PlyRectangle rect2( std::string("rect2"),
                      Vertex(0.,0.,1.),
                      Vertex(1.,0.,1.),
                      Vertex(1.,1.,1.),
                      Vertex(0.,1.,1.)
                    );
  PlyRectangle rect3( std::string("rect3"),
                      Vertex(0.,0.,2.),
                      Vertex(1.,0.,2.),
                      Vertex(1.,1.,2.),
                      Vertex(0.,1.,2.)
                    );
  PlyRectangle rect4( std::string("rect4"),
                      Vertex(-1., 0., 0.),
                      Vertex(0., -1., 0.),
                      Vertex(0., -1., 1.),
                      Vertex(-1., 0., 1.)
                    );
  
  Scene scene1("scene1");
  scene1.add(&rect1);
  scene1.add(&rect2);
  scene1.add(&rect3);

  Scene scene2("scene2");
  scene2.add(&rect4);
  
  Scene combined("combined");
  combined.add(&scene1);
  combined.add(&scene2);

  Iterator<PlyGeometry *> * it = combined.createIterator();
  for(it->first(); !it->isDone(); it->next()) {
    std::cout << it->currentItem()->name() << std::endl;
  }

  PlyWriter writer("test_PlyWriter_output.ply");
  writer.write(combined);
  writer.close();

  return 0;
}
