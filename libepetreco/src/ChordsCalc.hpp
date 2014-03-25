#ifndef CHORDSCALC_HPP
#define CHORDSCALC_HPP

template<typename Ray, typename Grid, typename Chord>
class ChordsCalc
{
  public:
   
    typedef          Ray                      Ray_t;
    typedef          Grid                     Grid_t;
    typedef          Chord                    Chord_t;
    typedef typename Ray_t::Vertex_t::Coord_t Coord_t;
        
    ChordsCalc();

    void getChords( Chord_t * a, Ray_t ray, Grid_t grid ); 

    int getNChords( Ray_t ray, Grid_t grid );
    
    
//  private:
    
    bool valid( Ray_t ray, Grid_t grid );
    
    //bool intersects( Ray_t ray, Grid_t grid, int dim );
    void getCrossesPlanes( Ray_t ray, Grid_t grid, bool * crosses );

    //bool intersectsAny( Ray_t ray, Grid_t grid );
    bool crossesAnyPlanes( Ray_t ray, Grid_t grid );
    
    Coord_t alphaFromId(
          int i, Ray_t ray, Grid_t grid, int dim );
    
    Coord_t phiFromAlpha(
          Coord_t alpha, Ray_t ray, Grid_t grid, int dim );
    
    //Coord_t getAlphaDimmin( Ray_t ray, Grid_t grid, int dim );
    void getAlphaDimmin( Ray_t ray, Grid_t grid, Coord_t * adimmin, bool * good );
    
    //Coord_t getAlphaDimmax( Ray_t ray, Grid_t grid, int dim );
    void getAlphaDimmax( Ray_t ray, Grid_t grid, Coord_t * adimmax, bool * good );
    
    //Coord_t getAlphaMin( Ray_t ray, Grid_t grid );
    void getAlphaMin( Ray_t ray, Grid_t grid, Coord_t * amin, bool * good );
    
    //Coord_t getAlphaMax( Ray_t ray, Grid_t grid );
    void getAlphaMax( Ray_t ray, Grid_t grid, Coord_t * amax, bool * good );
    
    int getIdDimmin( Ray_t ray, Grid_t grid, int dim );
    
    int getIdDimmax( Ray_t ray, Grid_t grid, int dim );
    
    void updateAlpha(
          Coord_t & alpha, Ray_t ray, Grid_t grid,
          int dim );
    
    void updateId( int & id, Ray_t ray, Grid_t grid, int dim );
};
#include "ChordsCalc.tpp"

#endif  // #ifndef CHORDSCALC_HPP
