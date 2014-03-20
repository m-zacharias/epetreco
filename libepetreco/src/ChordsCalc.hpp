#ifndef CHORDSCALC_HPP
#define CHORDSCALC_HPP

template<typename Ray, typename Grid, typename Chord>
class ChordsCalc
{
  public:
    
    typedef typename Ray::Vertex_t::Coord_t Coord_t;
    typedef Chord                  Chord_t;
        
    ChordsCalc();

    void getChords( Chord * a, Ray ray, Grid grid ); 

    int getNChords( Ray ray, Grid grid );
    
    
  private:
    
    bool valid( Ray ray, Grid grid );
    
    bool intersects( Ray ray, Grid grid, int dim );

    bool intersectsAny( Ray ray, Grid grid );
    
    typename Ray::Vertex_t::Coord_t alphaFromId(
          int i, Ray ray, Grid grid, int dim );
    
    typename Ray::Vertex_t::Coord_t phiFromAlpha(
          typename Ray::Vertex_t::Coord_t alpha, Ray ray, Grid grid, int dim );
    
    typename Ray::Vertex_t::Coord_t getAlphaDimmin(
          Ray ray, Grid grid, int dim );
    
    typename Ray::Vertex_t::Coord_t getAlphaDimmax(
          Ray ray, Grid grid, int dim );
    
    typename Ray::Vertex_t::Coord_t getAlphaMin( Ray ray, Grid grid );
    
    typename Ray::Vertex_t::Coord_t getAlphaMax( Ray ray, Grid grid );
    
    int getIdDimmin( Ray ray, Grid grid, int dim );
    
    int getIdDimmax( Ray ray, Grid grid, int dim );
    
    void updateAlpha(
          typename Ray::Vertex_t::Coord_t & alpha, Ray ray, Grid grid,
          int dim );
    
    void updateId( int & id, Ray ray, Grid grid, int dim );
};
#include "ChordsCalc.tpp"

#endif  // #ifndef CHORDSCALC_HPP
