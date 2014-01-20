#include "CompositePlyGeometry.hpp"
#include "PlyBox.hpp"
#include "PlyWriter.hpp"

class Scene : public CompositePlyGeometry
{
  public:
    
    Scene( std::string const name )
    : CompositePlyGeometry(name) {}
};




int main()
{
  Scene scene("scene");
  
  PlyBox box1("box1", Vertex(0.,  0.,  0.), 1., 1., 1.);
  PlyBox box2("box2", Vertex(1.5, 1.5, 0.), 0.5, 0.5, 0.5);
  PlyBox box3("box3", Vertex(0.,  2.,  0.), 1., 5., 1.);
  PlyBox box4("box4", Vertex(0.,  -1., 0.), 1., -5., 1.);

  scene.add(&box1); scene.add(&box2); scene.add(&box3); scene.add(&box4);

  PlyWriter writer("test_PlyBox_output.ply");
  writer.write(scene);
  writer.close();
  
  return 0;
}
