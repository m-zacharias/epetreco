#ifndef CUDAVECTOR_HPP
#define CUDAVECTOR_HPP

#include "Vector.hpp"

template<class TE, class TI>
class CudaVector : public Vector<TE>
{
  public:
    
    typedef TI internal_elem_t;
    
    
    CudaVector( int n );

    ~CudaVector();
    
    
    int get_n();
    
    void * data();
    
    TE get( int id );
    
    void set( int id, TE val );
    
    Vector<TE> * clone();
    
    void set_devi_data_changed();
    
    
  private:
    
    void update_devi_data();
    
    void update_host_data();
    
    
    TI * raw_host_;
    
    TI * raw_devi_;
    
    bool devi_data_changed_;
    
    bool host_data_changed_;
    
    int n_;
};
#include "CudaVector.tpp"

#endif  // #define CUDAVECTOR_HPP
