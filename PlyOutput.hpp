#ifndef PLYOUTPUT_HPP
#define PLYOUTPUT_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace PlyOutput
{
  typedef float coord_type;
  
  struct Vertex {
    public:
      
      coord_type x_, y_, z_;
      
      Vertex( void );
      
      Vertex( coord_type const x,
              coord_type const y,
              coord_type const z );
      
      Vertex( Vertex const & ori );
      
      ~Vertex( void );
      
      void operator=( Vertex const & ori );
  
      coord_type get_x( void ) const;
      
      coord_type get_y( void ) const;
      
      coord_type get_z( void ) const;
  };
  
  
  
  class Geometry {
    public:
      
      virtual int get_num_vertices( void ) = 0;
      
      virtual int get_num_faces( void ) const = 0;
      
      virtual std::string get_vertices_str( void ) = 0;
      
      virtual std::string get_faces_str( int & vertex_id ) = 0;
  };
  
  
  
  class Triangle : public Geometry {
    public:
      
      Vertex p0_, p1_, p2_;
      
      
      
    public:
      
      Triangle( void );
      
      Triangle( Vertex const & p0,
                Vertex const & p1,
                Vertex const & p2 );
      
      Triangle( Triangle const & ori );
      
      ~Triangle( void );
      
      void operator=( Triangle const rval );
      
      int get_num_vertices( void );
      
      int get_num_faces( void ) const;
      
      std::string get_vertices_str( void );
      
      std::string get_faces_str( int & );
  };
  
  
  
  class Rectangle : public Geometry {
    protected:
      
      Vertex p0_, p1_, p2_, p3_;
      
      
      
    public:
    
      Rectangle( void );
      
      Rectangle( Vertex const & p0,
                 Vertex const & p1,
                 Vertex const & p2,
                 Vertex const & p3 );
     
      Rectangle( Rectangle const & ori );
     
      Rectangle( Vertex const & p0,
                 coord_type const & lx,
                 coord_type const & ly );
  
      ~Rectangle( void );
      
      void operator=( Rectangle const rval );
          
      int get_num_vertices( void );
      
      int get_num_faces( void ) const;
      
      std::string get_vertices_str( void );
      
      std::string get_faces_str( int & vertex_id );
  };
  
  
  
  class Box : public Geometry {
    protected:
      
      Vertex p0_, p1_, p2_, p3_, p4_, p5_, p6_, p7_;
      
      
      
    public:
      
      Box( void );
      
      Box( Vertex const & p0,
           Vertex const & p1,
           Vertex const & p2,
           Vertex const & p3,
           Vertex const & p4,
           Vertex const & p5,
           Vertex const & p6,
           Vertex const & p7 );
      
      Box( Vertex const & p0,
           coord_type const & lx,
           coord_type const & ly,
           coord_type const & lz );
  
      Box( Box const & ori );
      
      ~Box( void );
      
      void operator=( Box const & rval );
      
      int get_num_vertices( void );
      
      int get_num_faces( void ) const;
      
      std::string get_vertices_str( void );
      
      std::string get_faces_str( int & vertex_id );
  };
  
  
  
  class Line : public Triangle {
    private:
      
      Vertex midpoint( Vertex const & a, Vertex const & b ) const;
      
      
      
    public:
    
      Line( void );
      
      Line( Vertex const & p0,
            Vertex const & p2 );
      
      Line( Line const & ori );
      
      ~Line( void );
  };
  
  
  
  
  class Scene {
    private:
      
      std::vector<Geometry *> index_;
      
      
      
    public:
      
      Scene( void );
      
      void add_Geometry( Geometry * geometry );
      
      int get_num_vertices( void ) const;
      
      int get_num_faces( void ) const;
      
      std::string get_vertices_str( void );
      
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
}

#endif  // #ifndef PLYOUTPUT_HPP
