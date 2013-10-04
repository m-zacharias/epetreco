#include "PlyWriter.hpp"

//~ std::string Geometry::get_triangles_str( int & vertex_id ) {
  //~ std::stringstream ss;
  //~ ss.str() = "";
  //~ 
  //~ Triangle * triangles = 0;
  //~ int num_triangles;
  //~ 
  //~ get_triangle_representation( triangles, num_triangles );
  //~ 
  //~ for( int i=0; i<num_triangles; i++ ) {
    //~ ss << triangles[i].get_triangles_str( vertex_id );
  //~ }
  //~ 
  //~ return ss.str();
//~ }

std::string Geometry::get_vertices_str( void ) {
  #ifdef DEBUG
  std::cout << "Geometry::get_vertices_str ..." << std::flush;
  #endif
  std::stringstream ss;
  ss.str() = "";
  
  Face * faces = 0;
  int num_faces;
  
  get_face_repr( faces, num_faces );
  
  for( int i=0; i<num_faces; i++ ) {
    std::cout << "(i,num_faces=" << i << "," << num_faces << ")" << std::flush;
    // !!! Can't dereference faces[i] because faces is a pointer to
    // !!! (virtual) base class.  Overload operator[] or define an
    // !!! get_index() function etc.
    ss << faces[i].get_vertices_str();
  }
  
  #ifdef DEBUG
  std::cout << " done" << std::endl;
  #endif
  return ss.str();
}

std::string Geometry::get_faces_str( int & vertex_id ) {
  std::stringstream ss;
  ss.str() = "";
  
  Face * faces = 0;
  int num_faces;
  
  get_face_repr( faces, num_faces );
  
  for( int i=0; i<num_faces; i++ ) {
    ss << faces[i].get_faces_str( vertex_id );
  }
  
  return ss.str();
}



void Face::get_face_repr( Face * & faces, int & num_faces ) {
  #ifdef DEBUG
  std::cout << "Face::get_face_repr ..." << std::flush;
  #endif
  faces = this;
  num_faces = 1;
  #ifdef DEBUG
  std::cout << " done" << std::endl;
  #endif
}



Triangle::Triangle( void ) {
  p0_ = new coord_type[3];
  p1_ = new coord_type[3];
  p2_ = new coord_type[3];
}

Triangle::Triangle( coord_type const * const p0,
          coord_type const * const p1,
          coord_type const * const p2 ) {
  p0_ = new coord_type[3];
  p1_ = new coord_type[3];
  p2_ = new coord_type[3];
  
  for( int i=0; i<3; i++ ) {
    p0_[i] = p0[i];
    p1_[i] = p1[i];
    p2_[i] = p2[i];
  }
}

Triangle::Triangle( Triangle const & ori ) {
  p0_ = new coord_type[3];
  p1_ = new coord_type[3];
  p2_ = new coord_type[3];
  
  for( int i=0; i<3; i++ ) {
    p0_[i] = ori.p0_[i];
    p1_[i] = ori.p1_[i];
    p2_[i] = ori.p2_[i];
  }
}

Triangle::~Triangle( void ) {
  delete[] p0_;
  delete[] p1_;
  delete[] p2_;
}

void Triangle::operator=( Triangle const rval ) {
  for( int i=0; i<3; i++ ) {
    p0_[i] = rval.p0_[i];
    p1_[i] = rval.p1_[i];
    p2_[i] = rval.p2_[i];
  }
}

int Triangle::get_num_faces( void ) const {
  return 1;
}

int Triangle::get_num_vertices( void ) {
  return 3;
}

//~ void Triangle::get_triangle_representation( Triangle * & triangle,
                                  //~ int & num_triangles ) {
  //~ num_triangles = get_num_faces();
  //~ triangle = this;
//~ }

std::string Triangle::get_vertices_str( void ) {
  std::stringstream ss;
  ss.str() = "";
  ss << p0_[0] << " " << p0_[1] << " " << p0_[2] 
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl
     << p1_[0] << " " << p1_[1] << " " << p1_[2]
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl
     << p2_[0] << " " << p2_[1] << " " << p2_[2]
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl;
  return ss.str();
}

std::string Triangle::get_triangles_str( int & vertex_id ) {
  std::stringstream ss;
  ss.str() = "";
  ss << "3 " << vertex_id << " " << vertex_id+1 << " " << vertex_id+2 << std::endl;
  vertex_id += 3;
  return ss.str();
}

std::string Triangle::get_faces_str( int & vertex_id ) {
  return get_triangles_str( vertex_id );
}



Rectangle::Rectangle( void )
//~ : tri_repr_(0), has_tri_repr_(false) {
{
  p0_ = new coord_type[3];
  p1_ = new coord_type[3];
  p2_ = new coord_type[3];
  p3_ = new coord_type[3];
}

Rectangle::Rectangle( coord_type const * const p0,
           coord_type const * const p1,
           coord_type const * const p2,
           coord_type const * const p3 )
//~ : tri_repr_(0), has_tri_repr_(false) {
{
  p0_ = new coord_type[3];
  p1_ = new coord_type[3];
  p2_ = new coord_type[3];
  p3_ = new coord_type[3];
  
  for( int i=0; i<3; i++ ) {
    p0_[i] = p0[i];
    p1_[i] = p1[i];
    p2_[i] = p2[i];
    p3_[i] = p3[i];
  }
}

Rectangle::~Rectangle( void ) {
  delete[] p0_;
  delete[] p1_;
  delete[] p2_;
  delete[] p3_;
  //~ delete[] tri_repr_;
}

void Rectangle::operator=( Rectangle const rval ) {
  for( int i=0; i<3; i++ ) {
    p0_[i] = rval.p0_[i];
    p1_[i] = rval.p1_[i];
    p2_[i] = rval.p2_[i];
    p3_[i] = rval.p3_[i];
  }
}

int Rectangle::get_num_vertices( void ) {
  return 4;
}

int Rectangle::get_num_faces( void ) const {
  return 1;
}

//~ void Rectangle::get_triangle_representation( Triangle * & triangles,
                                             //~ int & num_triangles ) {
  //~ if( !has_tri_repr_ ) {
    //~ tri_repr_ = new Triangle[2];
    //~ tri_repr_[0] = Triangle( p0_, p2_, p3_ );
    //~ tri_repr_[1] = Triangle( p2_, p0_, p1_ );
    //~ has_tri_repr_ = true;
  //~ }
  //~ 
  //~ triangles = tri_repr_;
  //~ num_triangles = 2;
//~ }

std::string Rectangle::get_vertices_str( void ) {
  #ifdef DEBUG
  std::cout << "Rectangle::get_vertices_str ..." << std::flush;
  #endif
  std::stringstream ss;
  ss.str() = "";
  
  ss << p0_[0] << " " << p0_[1] << " " << p0_[2] 
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl
     << p1_[0] << " " << p1_[1] << " " << p1_[2]
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl
     << p2_[0] << " " << p2_[1] << " " << p2_[2]
     << " " << 0 << " " << 0 << " " << 0 << " " << std::endl
     << p3_[0] << " " << p3_[1] << " " << p3_[2]
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



Box::Box( void )
: rect_repr_(0), has_rect_repr_(false) {
  p0_ = new coord_type[3];
  p1_ = new coord_type[3];
  p2_ = new coord_type[3];
  p3_ = new coord_type[3];
  p4_ = new coord_type[3];
  p5_ = new coord_type[3];
  p6_ = new coord_type[3];
  p7_ = new coord_type[3];
}
    
Box::Box( coord_type const * const p0,
     coord_type const * const p1,
     coord_type const * const p2,
     coord_type const * const p3,
     coord_type const * const p4,
     coord_type const * const p5,
     coord_type const * const p6,
     coord_type const * const p7 )
: rect_repr_(0), has_rect_repr_(false) {
  p0_ = new coord_type[3];
  p1_ = new coord_type[3];
  p2_ = new coord_type[3];
  p3_ = new coord_type[3];
  p4_ = new coord_type[3];
  p5_ = new coord_type[3];
  p6_ = new coord_type[3];
  p7_ = new coord_type[3];
  
  for( int i=0; i<3; i++ ) {
    p0_[i] = p0[i];
    p1_[i] = p1[i];
    p2_[i] = p2[i];
    p3_[i] = p3[i];
    p4_[i] = p4[i];
    p5_[i] = p5[i];
    p6_[i] = p6[i];
    p7_[i] = p7[i];
  }
}

Box::~Box( void ) {
  delete[] p0_;
  delete[] p1_;
  delete[] p2_;
  delete[] p3_;
  delete[] p4_;
  delete[] p5_;
  delete[] p6_;
  delete[] p7_;
  delete[] rect_repr_;
}

int Box::get_num_faces( void ) const {
  return 6;
}

int Box::get_num_vertices( void ) {
  return 18;
}

//~ void Box::get_triangle_representation( Triangle * & triangles,
                                       //~ int & num_triangles ) {
  //~ if( !has_tri_repr_ ) {
    //~ tri_repr_ = new Triangle[12];
    //~ tri_repr_[0] = Triangle( p0_, p7_, p4_ );
    //~ tri_repr_[1] = Triangle( p7_, p0_, p3_ );
    //~ tri_repr_[2] = Triangle( p1_, p6_, p5_ );
    //~ tri_repr_[3] = Triangle( p6_, p1_, p2_ );
    //~ tri_repr_[4] = Triangle( p1_, p4_, p5_ );
    //~ tri_repr_[5] = Triangle( p4_, p1_, p0_ );
    //~ tri_repr_[6] = Triangle( p2_, p7_, p6_ );
    //~ tri_repr_[7] = Triangle( p7_, p2_, p3_ );
    //~ tri_repr_[8] = Triangle( p0_, p2_, p3_ );
    //~ tri_repr_[9] = Triangle( p2_, p0_, p1_ );
    //~ tri_repr_[10] = Triangle( p4_, p6_, p7_ );
    //~ tri_repr_[11] = Triangle( p6_, p4_, p5_ );
    //~ has_tri_repr_ = true;
  //~ }
  //~ 
  //~ triangles = tri_repr_;
  //~ num_triangles = 12;
//~ }

void Box::get_face_repr( Face * & faces, int & num_faces ) {
  #ifdef DEBUG
  std::cout << "Box::get_face_repr ..." << std::flush;
  #endif
  if( !has_rect_repr_ ) {
    rect_repr_ = new Rectangle[6];
    rect_repr_[0] = Rectangle( p0_, p3_, p7_, p4_ );
    rect_repr_[1] = Rectangle( p1_, p2_, p6_, p5_ );
    rect_repr_[2] = Rectangle( p1_, p0_, p4_, p5_ );
    rect_repr_[3] = Rectangle( p2_, p3_, p7_, p6_ );
    rect_repr_[4] = Rectangle( p0_, p1_, p2_, p3_ );
    rect_repr_[5] = Rectangle( p4_, p5_, p6_, p7_ );
    has_rect_repr_ = true;
  }
  std::cout << "(" << rect_repr_[1].get_num_vertices() << ")" << std::flush;
  
  faces = static_cast<Face *>( rect_repr_ );
  num_faces = 6;
  #ifdef DEBUG
  std::cout << " done" << std::endl;
  #endif
}



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

//~ std::string Scene::get_triangles_str( void ) {
  //~ std::stringstream ss;
  //~ ss.str() = "";
  //~ 
  //~ int vertex_id = 0;
  //~ std::vector<Geometry *>::const_iterator it = index_.begin();
  //~ while( it != index_.end() ) {
    //~ ss << (*it)->get_triangles_str( vertex_id );
    //~ it++;
  //~ }
  //~ 
  //~ return ss.str();
//~ }

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
