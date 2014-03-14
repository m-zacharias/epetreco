#include "PlyRectangle.hpp"
#include "CompositePlyGeometry.hpp"
#include "PlyWriter.hpp"
#include "Iterator.hpp"
#include "TemplateVertex.hpp"

#include <iostream>


class Scene : public CompositePlyGeometry
{
  public:
    
    Scene( std::string const name )
    : CompositePlyGeometry(name) {}
};


typedef double                     CoordType;
typedef TemplateVertex<CoordType>  VertexType;



int main()
{
  PlyRectangle<VertexType> rect1( std::string("rect1"),
                      VertexType(0.,0.,0.),
                      VertexType(1.,0.,0.),
                      VertexType(1.,1.,0.),
                      VertexType(0.,1.,0.)
                    );
  PlyRectangle<VertexType> rect2( std::string("rect2"),
                      VertexType(0.,0.,1.),
                      VertexType(1.,0.,1.),
                      VertexType(1.,1.,1.),
                      VertexType(0.,1.,1.)
                    );
  PlyRectangle<VertexType> rect3( std::string("rect3"),
                      VertexType(0.,0.,2.),
                      VertexType(1.,0.,2.),
                      VertexType(1.,1.,2.),
                      VertexType(0.,1.,2.)
                    );
  PlyRectangle<VertexType> rect4( std::string("rect4"),
                      VertexType(-1., 0., 0.),
                      VertexType(0., -1., 0.),
                      VertexType(0., -1., 1.),
                      VertexType(-1., 0., 1.)
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
