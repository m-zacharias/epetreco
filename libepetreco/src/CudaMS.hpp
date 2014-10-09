#ifndef CUDAMS
#define CUDAMS

template<typename T, typename ConcreteMeasurementSetup>
class CudaMS
{
  public:
    
    CudaMS(
          T   pos0x, T   pos1x,
          int na,    int n0z,   int n0y,  int n1z, int n1y,
          T   da,    T   segx,  T   segy, T   segz )
    {
      // Allocate host memory
      _data_host = new ConcreteMeasurementSetup(
            pos0x, pos1x, na, n0z, n0y, n1z, n1y, da, segx, segy, segz);
      _host_data_changed = true;
      
      // Allocate device memory
      cudaError_t status;
      status =
            cudaMalloc((void**)&_data_devi, sizeof(ConcreteMeasurementSetup));
      _devi_data_changed = false; 
      if(status != cudaSuccess)
      {
        std::cerr << "CudaMS::CudaMS(...) : cudaMalloc(...) failed" << std::endl;
        exit(EXIT_FAILURE);
      }
      _devi_data_changed = false;
    }

    ~CudaMS()
    {
      delete _data_host;
      cudaFree(_data_devi);
    }

    ConcreteMeasurementSetup * deviRepr()
    {
      if(_host_data_changed)
        update_devi_data();
      
      _devi_data_changed = true;
      return _data_devi;
    }

    ConcreteMeasurementSetup * hostRepr()
    {
      if(_devi_data_changed)
        update_host_data();
      
      _host_data_changed = true;
      return _data_host;
    }


  private:
    
    ConcreteMeasurementSetup * _data_host;
    
    ConcreteMeasurementSetup * _data_devi;

    bool _host_data_changed;

    bool _devi_data_changed;

    void update_host_data()
    {
      cudaError_t status;
      status =
            cudaMemcpy(_data_host, _data_devi, sizeof(ConcreteMeasurementSetup),
                       cudaMemcpyDeviceToHost);
      if(status != cudaSuccess)
      {
        std::cerr << "CudaMS::update_host_data() : cudaMemcpy(...) failed"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      _devi_data_changed = false;
    }

    void update_devi_data()
    {
      cudaError_t status;
      status =
            cudaMemcpy(_data_devi, _data_host, sizeof(ConcreteMeasurementSetup),
                       cudaMemcpyHostToDevice);
      if(status != cudaSuccess)
      {
        std::cerr << "CudaMS::update_devi_data() : cudaMemcpy(...) failed"
                  << std::endl;
        exit(EXIT_FAILURE);
      }
      _host_data_changed = false;
    }
};

#endif  // #define CUDAMS
