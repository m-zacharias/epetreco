#ifndef CUDAVG
#define CUDAVG

template<typename T, typename ConcreteVoxelGrid>
class CudaVG
{
  public:
    
    CudaVG( T const   gridO0, T const   gridO1, T const   gridO2,
            T const   gridD0, T const   gridD1, T const   gridD2,
            int const gridN0, int const gridN1, int const gridN2 )
    {
      // Allocate host memory
      _data_host = new ConcreteVoxelGrid(gridO0, gridO1, gridO2,
                                         gridD0, gridD1, gridD2,
                                         gridN0, gridN1, gridN2);
      _host_data_changed = true;
      
      // Allocate device memory
      cudaError_t status;
      status =
            cudaMalloc((void**)&_data_devi, sizeof(ConcreteVoxelGrid));
      if(status != cudaSuccess)
      {
        std::cerr << "CudaVG::CudaVG(...) : cudaMalloc(...) failed" << std::endl;
        exit(EXIT_FAILURE);
      }
      _devi_data_changed = false;
    }

    ~CudaVG()
    {
      delete _data_host;
      cudaFree(_data_devi);
    }

    ConcreteVoxelGrid * deviRepr()
    {
      if(_host_data_changed)
        update_devi_data();
      
      _devi_data_changed = true;
      return _data_devi;
    }

    ConcreteVoxelGrid * hostRepr()
    {
      if(_devi_data_changed)
        update_host_data();
      
      _host_data_changed = true;
      return _data_host;
    }


  protected:
    
    ConcreteVoxelGrid * _data_host;
    
    ConcreteVoxelGrid * _data_devi;

    bool _host_data_changed;

    bool _devi_data_changed;

    void update_host_data()
    {
      cudaError_t status;
      status =
            cudaMemcpy(this->_data_host, this->_data_devi, sizeof(ConcreteVoxelGrid),
                       cudaMemcpyDeviceToHost);
      if(status != cudaSuccess)
      {
        std::cerr << "CudaVG::update_host_data() : cudaMemcpy(...) failed: "
                  << cudaGetErrorString(status)
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      _devi_data_changed = false;
    }

    void update_devi_data()
    {
      cudaError_t status;
      status =
            cudaMemcpy(_data_devi, _data_host, sizeof(ConcreteVoxelGrid),
                       cudaMemcpyHostToDevice);
      if(status != cudaSuccess)
      {
        std::cerr << "CudaVG::update_devi_data() : cudaMemcpy(...) failed"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      _host_data_changed = false;
    }
};

#endif  // #define CUDAVG
