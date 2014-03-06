#ifndef CUDAVECTOR_HPP
#define CUDAVECTOR_HPP

#include "Vector.hpp"



template<typename TE, typename TI>
struct CudaVectorTraits
{
  public:
    
    typedef TE                elem_t;
    typedef TE                external_elem_t;
    typedef TI                internal_elem_t;
};



template<class TE, class TI>
class CudaVector : public Vector<CudaVector<TE,TI>, CudaVectorTraits<TE,TI> >
{
  public:
    
    typedef typename CudaVectorTraits<TE, TI>::elem_t          elem_t;
    typedef typename CudaVectorTraits<TE, TI>::external_elem_t external_elem_t;
    typedef typename CudaVectorTraits<TE, TI>::internal_elem_t internal_elem_t;

    CudaVector( int n );

    ~CudaVector();
    
    
    int get_n();
    
    void * data();
    
    TE get( int id );
    
    void set( int id, TE val );
    
    CudaVector<TE, TI> * clone();
    
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
