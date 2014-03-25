#ifndef DEVELCHANNEL_HPP
#define DEVELCHANNEL_HPP

#include "DevelRay.hpp"
#include "Ply.hpp"

// Define channel type
class DevelChannel
{
  public:
    
    friend class MySetup;

    typedef          DevelRay                  Ray_t;
    typedef typename Ray_t::Vertex_t::Coord_t  Coord_t;

    class PlyRepr : public CompositePlyGeometry
    {
      public:
        
        // Default Constructor
        PlyRepr()
        : CompositePlyGeometry(std::string("")) {}
    
        // Copy Constructor
        PlyRepr( PlyRepr const & ori )
        : CompositePlyGeometry(std::string(""))
        {
          _geometryList = ori._geometryList;
        }
    };


  private:
    
    int _angle;
    Coord_t _pos0[3]; 
    Coord_t _pos1[3];
//    int _nrays;
//    DevelRay * _rays;

//    bool _updateRayMemSize( int nrays );

   /** 
    * @brief Write the transformation matrix: {random(0.,1.)**3} ->
    * {random point within detector segment} into mem_trafo.
    *
    * The transformation consists of 4 consecutive steps:
    * - translate by (-0.5,-0.5,-0.5) (accounts for segment position==center of
    *   segment box)
    * - scale by (edgelength_x,edgelength_y,edgelength_z)
    * - translate by detector segment basic position
    * - rotate by angle of channel
    * 
    * This can, in homogeneous coordinates, be perceived as:
    *   (R_phi * T_segpos * S_edgelengths * T_{-.5}) * (random_{(0..1)^3},1)
    * 
    * Which can be expressed as:
    *   B_transform                                  * (random_{(0..1)^3},1)
    * 
    * B_transform is referred to as transformation matrix.
    */
    void _trafo(
          Coord_t * const mem_trafo, Coord_t * const edges,
          Coord_t * const pos, Coord_t const sin, Coord_t const cos );


  public:
    
    /**
     * Write central position of detector 0 segment in basic position (i.e. for
     * angle = 0 degrees) into mem_pos0
     */
    void getPos0( Coord_t * mem_pos0 );
    
    /**
     * Write central position of detector 1 segment in basic position (i.e. for
     * angle = 0 degrees) into mem_pos1
     */
    void getPos1( Coord_t * mem_pos1 );

    /**
     * Return angular positon of the detector heads in degrees
     */
    int getAngle();
    
    /**
     * Write the detectors segments' edge lengths into mem_edges
     */
    void getEdges( Coord_t * mem_edges );

//    /**
//     * Return number of rays
//     */
//    int getNRays( void ) const;
    
    /**
     * Write a number of randomly chosen rays representing the channel into
     * rays.
     */
    void createRays( Ray_t * rays, int nRays ); 
    
//    /**
//     * Write a number of randomly chosen rays representing the channel into
//     * mem_rays.
//     */
//    void setRays( int nrays );

    PlyRepr getPlyRepr();

    // Constructor
    DevelChannel( int angle, Coord_t * pos0, Coord_t * pos1 );

    // Default Constructor
    DevelChannel( void );

    // Copy Constructor
    DevelChannel( DevelChannel const & ori );

    // Copy Assignment
    void operator=( DevelChannel const & ori );
};
#include "DevelChannel.tpp"

#endif  // #ifndef DEVELCHANNEL_HPP
