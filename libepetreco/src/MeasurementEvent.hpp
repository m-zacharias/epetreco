/** @file MeasurementEvent.hpp */
#ifndef MEASUREMENTEVENT
#define MEASUREMENTEVENT

template<typename T>
struct MeasurementEvent
{
  T   _value;
  int _channel;
  
  __host__ __device__
  MeasurementEvent()
  : _value(0.), _channel(-1) {}

  __host__ __device__
  MeasurementEvent( T value_, int channel_)
  : _value(value_), _channel(channel_) {}

  __host__ __device__
  MeasurementEvent( MeasurementEvent<T> const & ori )
  {
    _value   = ori._value;
    _channel = ori._channel;
  }
  
  __host__ __device__
  ~MeasurementEvent()
  {}

  __host__ __device__
  void operator=( MeasurementEvent<T> const & rhs )
  {
    _value   = rhs._value;
    _channel = rhs._channel;
  }

  __host__ __device__
  T value() const
  {
    return _value;
  }

  __host__ __device__
  int channel() const
  {
    return _channel;
  }
};

#endif  // #define MEASUREMENTEVENT

