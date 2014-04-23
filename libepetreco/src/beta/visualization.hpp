#include "Ply.hpp"
#include "TemplateVertex.hpp"
typedef TemplateVertex<val_t> Vertex;

class BetaPlyGrid : public PlyGrid<Vertex>
{
  public:
    
    enum OriginCMode
    {
      AT_ORIGIN,
    };

    enum CenterCMode
    {
      AT_CENTER,
    };

    BetaPlyGrid(
          std::string const name,
          val_t const * const gridO,
          val_t const * const gridD,
          int const * const gridN )
    : PlyGrid<Vertex>(name,
                      Vertex(gridO[0], gridO[1], gridO[2]),
                      gridN[0]+1, gridN[1]+1, gridN[2]+1,
                      gridD[0], gridD[1], gridD[2]) {}
    
    BetaPlyGrid(
          std::string const name,
          val_t const * const gridAt,
          val_t const * const gridD,
          int const * const gridN,
          OriginCMode cmode )
    : PlyGrid<Vertex>(name,
                      Vertex(gridAt[0], gridAt[1], gridAt[2]),
                      gridN[0]+1, gridN[1]+1, gridN[2]+1,
                      gridD[0], gridD[1], gridD[2]) {}
    
    BetaPlyGrid(
          std::string const name,
          val_t const * const gridAt,
          val_t const * const gridD,
          int const * const gridN,
          CenterCMode cmode )
    : PlyGrid<Vertex>(name,
                      Vertex(gridAt[0]-0.5*gridN[0]*gridD[0],
                             gridAt[1]-0.5*gridN[1]*gridD[1],
                             gridAt[2]-0.5*gridN[2]*gridD[2]),
                      gridN[0]+1, gridN[1]+1, gridN[2]+1,
                      gridD[0], gridD[1], gridD[2]) {}
};

class BetaCompositePlyGeom : public CompositePlyGeometry
{
  public:
    
    BetaCompositePlyGeom( std::string const name )
    : CompositePlyGeometry(name) {}
};

class BetaPlyLine : public PlyLine<Vertex>
{
  public:
    
    BetaPlyLine()
    : PlyLine("", Vertex(0,0,0), Vertex(0,0,0)) {}

    BetaPlyLine(
          std::string const name,
          val_t const * const ray)
    : PlyLine(name,
              Vertex(ray[0], ray[1], ray[2]),
              Vertex(ray[3], ray[4], ray[5])) {}
};
