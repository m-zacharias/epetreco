#ifndef PLYBOX_HPP
#define PLYBOX_HPP

#include "PlyGeometry.hpp"

/* A Leaf class */
class PlyBox : public PlyGeometry
{
  public:
    
    /* Constructor */
    PlyBox( std::string const, Vertex const, coord_type const, coord_type const,
            coord_type const );
    
    /* Destructor */
    virtual ~PlyBox();


    /* Get number of vertices */
    virtual int numVertices();
    
    /* Get number of faces */
    virtual int numFaces();
    
    /* Get vertices string */
    virtual std::string verticesStr();
    
    /* Get faces string */
    virtual std::string facesStr();


    Vertex & p0();

    Vertex & p1();

    Vertex & p2();

    Vertex & p3();

    Vertex & p4();
    
    Vertex & p5();
    
    Vertex & p6();
    
    Vertex & p7();


  protected:
    
    /* Get number of faces, intermediate call */
    virtual std::string facesStr( int & );


  private:
    
    Vertex _p0, _p1, _p2, _p3, _p4, _p5, _p6, _p7;
};

#endif  // #define PLYBOX_HPP
