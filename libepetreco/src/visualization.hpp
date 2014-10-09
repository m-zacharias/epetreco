#include "Ply.hpp"
#include "TemplateVertex.hpp"

//typedef TemplateVertex<val_t> Vertex;

template<typename T>
class BetaPlyGrid : public PlyGrid<TemplateVertex<T> >
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
          T const * const gridO,
          T const * const gridD,
          int const * const gridN )
    : PlyGrid<TemplateVertex<T> >(name,
                                  TemplateVertex<T>(gridO[0], gridO[1],
                                                    gridO[2]),
                                  gridN[0]+1, gridN[1]+1, gridN[2]+1,
                                  gridD[0], gridD[1], gridD[2]) {}
    
    BetaPlyGrid(
          std::string const name,
          T const * const gridAt,
          T const * const gridD,
          int const * const gridN,
          OriginCMode cmode )
    : PlyGrid<TemplateVertex<T> >(name,
                                  TemplateVertex<T>(gridAt[0], gridAt[1],
                                                    gridAt[2]),
                                  gridN[0]+1, gridN[1]+1, gridN[2]+1,
                                  gridD[0], gridD[1], gridD[2]) {}
    
    BetaPlyGrid(
          std::string const name,
          T const * const gridAt,
          T const * const gridD,
          int const * const gridN,
          CenterCMode cmode )
    : PlyGrid<TemplateVertex<T> >(name,
                                  TemplateVertex<T>(
                                         gridAt[0]-0.5*gridN[0]*gridD[0],
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

template<typename T>
class BetaPlyLine : public PlyLine<TemplateVertex<T> >
{
  public:
    
    BetaPlyLine()
    : PlyLine<TemplateVertex<T> >("",
                                  TemplateVertex<T>(0,0,0),
                                  TemplateVertex<T>(0,0,0)) {}

    BetaPlyLine(
          std::string const name,
          T const * const ray)
    : PlyLine<TemplateVertex<T> >(name,
                                  TemplateVertex<T>(ray[0], ray[1], ray[2]),
                                  TemplateVertex<T>(ray[3], ray[4], ray[5])) {}
};
