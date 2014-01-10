#include "Siddon.hpp"
#include <iostream>

using namespace Siddon;

int main( void ) {
    coordRay  ray(  coordVector(2.5,2.5,2.5),
                    coordVector(1.5,1.5,1.5)
              );

    coordGrid grid( coordVector(1.,1.,1.),
                    coordVector(1.,1.,1.),
                    Vector<int>(3,3,3) );

    std::cout << "alpha_xmin: " << get_alpha_dimmin(0,ray,grid) << std::endl
              << "alpha_min: "  << get_alpha_min(ray,grid)      << std::endl
              << "i_xmin: "     << get_i_dimmin(0,ray,grid)     << std::endl;
    std::cout << "alpha_xmax: " << get_alpha_dimmax(0,ray,grid) << std::endl
              << "alpha_max: "  << get_alpha_max(ray,grid)      << std::endl
              << "i_xmax: "     << get_i_dimmax(0,ray,grid)     << std::endl;

    return 0;
}
