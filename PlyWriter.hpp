#ifndef PLYWRITER_HPP
#define PLYWRITER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

typedef float coord_type;



class Face;



class Geometry {
  public:
    
    virtual int get_num_faces( void ) const = 0;
    
    virtual int get_num_vertices( void ) = 0;
    
    virtual void get_face_repr( Face * &, int & ) = 0;
    
    virtual std::string get_vertices_str( void );
    
    //~ virtual std::string get_triangles_str( int & vertex_id );
    
    virtual std::string get_faces_str( int & vertex_id );
};



class Face : public Geometry {
  public:
    
    void get_face_repr( Face * &, int & );
};



class Triangle : public Face {
  private:
    
    coord_type * p0_, * p1_, * p2_;
    
  public:
    
    Triangle( void );
    
    Triangle( coord_type const * const p0,
              coord_type const * const p1,
              coord_type const * const p2 );
    
    Triangle( Triangle const & ori );
    
    ~Triangle( void );
    
    void operator=( Triangle const rval );
    
    int get_num_faces( void ) const;
    
    int get_num_vertices( void );
    
    //~ void get_triangle_representation( Triangle * & triangle,
                                      //~ int & num_triangles );
    
    std::string get_vertices_str( void );
    
    std::string get_faces_str( int & );
    
    std::string get_triangles_str( int & );
};



class Rectangle : public Face {
  private:
    
    coord_type * p0_, * p1_, * p2_, * p3_;
    
    //~ Triangle * tri_repr_;
    //~ 
    //~ bool has_tri_repr_;
    
    
    
  public:
  
    Rectangle( void );
    
    Rectangle( coord_type const * const p0,
               coord_type const * const p1,
               coord_type const * const p2,
               coord_type const * const p3 );
    
    ~Rectangle( void );
    
    void operator=( Rectangle const rval );
        
    int get_num_faces( void ) const;
    
    int get_num_vertices( void );
    
    //~ void get_triangle_representation( Triangle * & triangles,
                                      //~ int & num_triangles );
    
    std::string get_vertices_str( void );
    
    std::string get_faces_str( int & vertex_id );
};



class Box : public Geometry {
  private:
    
    coord_type * p0_, * p1_, * p2_, * p3_, * p4_, * p5_, * p6_, * p7_;
    
    Rectangle * rect_repr_;
    
    bool has_rect_repr_;
    
    
    
  public:
    
    Box( void );
    
    Box( coord_type const * const p0,
         coord_type const * const p1,
         coord_type const * const p2,
         coord_type const * const p3,
         coord_type const * const p4,
         coord_type const * const p5,
         coord_type const * const p6,
         coord_type const * const p7 );
    
    ~Box( void );
    
    int get_num_faces( void ) const;
    
    int get_num_vertices( void );
    
    //~ void get_triangle_representation( Triangle * &,
                                              //~ int & );
    void get_face_repr( Face * & faces, int & num_faces );
};



class Scene {
  private:
    
    std::vector<Geometry *> index_;
    
    
    
  public:
    
    Scene( void );
    
    void add_Geometry( Geometry * geometry );
    
    int get_num_faces( void ) const;
    
    int get_num_vertices( void ) const;
    
    std::string get_vertices_str( void );
    
    //~ std::string get_triangles_str( void );
    
    std::string get_faces_str( void );
};



class PlyWriter {
  private:
    
    std::ofstream out_;
    
    void write_header( Scene const & scene );
  
  
  
  public:
    
    PlyWriter( void );
    
    PlyWriter( std::string const & fn );
    
    ~PlyWriter( void );
    
    void write( Scene & scene );
    
    void close( void );
};

#endif  // #ifndef PLYWRITER_HPP
