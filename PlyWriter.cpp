#include "PlyWriter.hpp"

Vertex::Vertex( void ) {}

Vertex::Vertex( coord_type const x,
                coord_type const y,
                coord_type const z )
: x_(x), y_(y), z_(z) {}

Vertex::Vertex( Vertex const & ori )
: x_(ori.x_), y_(ori.y_), z_(ori.z_) {}

Vertex::~Vertex( void ) {}

void Vertex::operator=( Vertex const & ori ) {
  x_ = ori.x_;
  y_ = ori.y_;
  z_ = ori.z_;
}



Triangle::Triangle( void ) {}

Triangle::Triangle( Vertex const & p0,
                    Vertex const & p1,
                    Vertex const & p2 ) 
: p0_(p0), p1_(p1), p2_(p2) {}

Triangle::Triangle( Triangle const & ori )
: p0_(ori.p0_), p1_(ori.p1_), p2_(ori.p2_) {}

Triangle::~Triangle( void ) {}

void Triangle::operator=( Triangle const rval ) {
  p0_ = rval.p0_;
  p1_ = rval.p1_;
  p2_ = rval.p2_;
}

int Triangle::get_num_faces( void ) const {
  return 1;
}

int Triangle::get_num_vertices( void ) {
  return 3;
}

std::string Triangle::get_vertices_str( void ) {
  std::stringstream ss;
  ss.str() = "";
  ss << p0_.x_ << " " << p0_.y_ << " " << p0_.z_ 
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl
     << p1_.x_ << " " << p1_.y_ << " " << p1_.z_
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl
     << p2_.x_ << " " << p2_.y_ << " " << p2_.z_
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl;
  return ss.str();
}

std::string Triangle::get_faces_str( int & vertex_id ) {
  std::stringstream ss;
  ss.str() = "";
  ss << "3 " << vertex_id << " " << vertex_id+1 << " " << vertex_id+2 << std::endl;
  vertex_id += 3;
  return ss.str();
}



Rectangle::Rectangle( void ) {}

Rectangle::Rectangle( Vertex const & p0,
                      Vertex const & p1,
                      Vertex const & p2,
                      Vertex const & p3 )
: p0_(p0), p1_(p1), p2_(p2), p3_(p3) {}

Rectangle::~Rectangle( void ) {}

void Rectangle::operator=( Rectangle const rval ) {
  p0_ = rval.p0_;
  p1_ = rval.p1_;
  p2_ = rval.p2_;
  p3_ = rval.p3_;
}

int Rectangle::get_num_vertices( void ) {
  return 4;
}

int Rectangle::get_num_faces( void ) const {
  return 1;
}

std::string Rectangle::get_vertices_str( void ) {
  #ifdef DEBUG
  std::cout << "Rectangle::get_vertices_str ..." << std::flush;
  #endif
  std::stringstream ss;
  ss.str() = "";
  
  ss << p0_.x_ << " " << p0_.y_ << " " << p0_.z_ 
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl
     << p1_.x_ << " " << p1_.y_ << " " << p1_.z_
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl
     << p2_.x_ << " " << p2_.y_ << " " << p2_.z_
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl
     << p3_.x_ << " " << p3_.y_ << " " << p3_.z_
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl;
  
  #ifdef DEBUG
  std::cout << " done" << std::endl;
  #endif
  return ss.str();
}

std::string Rectangle::get_faces_str( int & vertex_id ) {
  std::stringstream ss;
  ss.str() = "";
  
  ss << "4 "
     << vertex_id   << " " << vertex_id+1 << " "
     << vertex_id+2 << " " << vertex_id+3 << std::endl;
  vertex_id += 4;
  
  return ss.str();
}



Box::Box( void ) {}
    
Box::Box( Vertex const & p0,
          Vertex const & p1,
          Vertex const & p2,
          Vertex const & p3,
          Vertex const & p4,
          Vertex const & p5,
          Vertex const & p6,
          Vertex const & p7 )
: p0_(p0), p1_(p1), p2_(p2), p3_(p3),
  p4_(p4), p5_(p5), p6_(p6), p7_(p7) {}

Box::Box( Box const & ori )
: p0_(ori.p0_), p1_(ori.p1_), p2_(ori.p2_), p3_(ori.p3_),
  p4_(ori.p4_), p5_(ori.p5_), p6_(ori.p6_), p7_(ori.p7_) {}

Box::~Box( void ) {}

int Box::get_num_vertices( void ) {
  return 24;
}

int Box::get_num_faces( void ) const {
  return 6;
}

std::string Box::get_vertices_str( void ) {
  std::stringstream ss;
  ss.str() = "";
  
  ss << Rectangle( p0_, p3_, p7_, p4_ ).get_vertices_str()
     << Rectangle( p1_, p2_, p6_, p5_ ).get_vertices_str()
     << Rectangle( p1_, p0_, p4_, p5_ ).get_vertices_str()
     << Rectangle( p2_, p3_, p7_, p6_ ).get_vertices_str()
     << Rectangle( p0_, p1_, p2_, p3_ ).get_vertices_str()
     << Rectangle( p4_, p5_, p6_, p7_ ).get_vertices_str();
  
  return ss.str();
}

std::string Box::get_faces_str( int & vertex_id ) {
  std::stringstream ss;
  ss.str() = "";
  
  ss << Rectangle( p0_, p3_, p7_, p4_ ).get_faces_str( vertex_id )
     << Rectangle( p1_, p2_, p6_, p5_ ).get_faces_str( vertex_id )
     << Rectangle( p1_, p0_, p4_, p5_ ).get_faces_str( vertex_id )
     << Rectangle( p2_, p3_, p7_, p6_ ).get_faces_str( vertex_id )
     << Rectangle( p0_, p1_, p2_, p3_ ).get_faces_str( vertex_id )
     << Rectangle( p4_, p5_, p6_, p7_ ).get_faces_str( vertex_id );
  
  return ss.str();
}



Vertex Line::midpoint( Vertex const & a, Vertex const & b ) const {
  return Vertex( 0.5*(a.x_+b.x_),
                 0.5*(a.y_+b.y_),
                 0.5*(a.z_+b.z_) );
} 

Line::Line( void ) {
  #ifdef DEBUG
  std::cout << "Line::Line() ..." << std::flush;
  std::cout << " done" << std::endl;
  #endif
}

Line::Line( Vertex const & p0,
            Vertex const & p2 )
: Triangle(p0, midpoint(p0,p2), p2) {}

Line::Line( Line const & ori )
: Triangle(ori) {}

Line::~Line( void ) {}



Scene::Scene( void )
: index_(std::vector<Geometry *>()) {}

void Scene::add_Geometry( Geometry * geometry ) {
  index_.push_back( geometry );
}

int Scene::get_num_faces( void ) const {
  int temp_num = 0;
  
  std::vector<Geometry *>::const_iterator it = index_.begin();
  while( it != index_.end() ) {
    temp_num += (*it)->get_num_faces();
    it++;
  }
  
  return temp_num;
}

int Scene::get_num_vertices( void ) const {
  int temp_num = 0;
  
  std::vector<Geometry *>::const_iterator it = index_.begin() ;
  while( it != index_.end() ) {
    temp_num += (*it)->get_num_vertices();
    it++;
  }
  
  return temp_num;
}

std::string Scene::get_vertices_str( void ) {
  #ifdef DEBUG
  std::cout << "Scene::get_vertices_str ..." << std::flush;
  #endif
  std::stringstream ss;
  ss.str() = "";
  
  #ifdef DEBUG
  std::cout << std::endl << "    loop index_ ..." << std::flush;
  #endif
  std::vector<Geometry *>::const_iterator it = index_.begin();
  while( it != index_.end() ) {
    ss << (*it)->get_vertices_str();
    it++;
  }
  #ifdef DEBUG
  std::cout << "    done" << std::endl;
  #endif
  
  #ifdef DEBUG
  std::cout << " done" << std::endl;
  #endif
  return ss.str();
}

std::string Scene::get_faces_str( void ) {
  #ifdef DEBUG
  std::cout << "Scene::get_faces_str ..." << std::flush;
  #endif
  std::stringstream ss;
  ss.str() = "";
  
  int vertex_id = 0;
  std::vector<Geometry *>::const_iterator it = index_.begin();
  while( it != index_.end() ) {
    ss << (*it)->get_faces_str( vertex_id );
    it++;
  }
  
  #ifdef DEBUG
  std::cout << " done" << std::endl;
  #endif
  return ss.str();
}



void PlyWriter::write_header( Scene const & scene ) {
  out_ << "ply" << std::endl
  //~ std::cout << "ply" << std::endl
       << "format ascii 1.0" << std::endl
       << "comment Created by object PlyWriter" << std::endl
       << "element vertex "
       << scene.get_num_vertices() << std::endl
       << "property float x" << std::endl
       << "property float y" << std::endl
       << "property float z" << std::endl
       << "property float nx" << std::endl
       << "property float ny" << std::endl
       << "property float nz" << std::endl
       << "element face "
       << scene.get_num_faces() << std::endl
       << "property list uchar uint vertex_indices" << std::endl
       << "end_header" << std::endl;
}

PlyWriter::PlyWriter( void ) {}

PlyWriter::PlyWriter( std::string const & fn ) {
  out_.open( fn.c_str() );
}

PlyWriter::~PlyWriter( void ) {
  out_.close();
}

void PlyWriter::write( Scene & scene ) {
  #ifdef DEBUG
  std::cout << std::endl << "    PlyWriter::write_header ..." << std::flush;
  #endif
  write_header( scene );
  #ifdef DEBUG
  std::cout << " done" << std::endl;
  #endif
  out_ << scene.get_vertices_str();
  out_ << scene.get_faces_str();
  //~ std::cout << scene.get_vertices_str();
  //~ std::cout << scene.get_faces_str();
}

void PlyWriter::close( void ) {
  out_.close();
}
