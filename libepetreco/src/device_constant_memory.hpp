/** @file device_constant_memory.hpp */
#ifndef DEVICE_CONSTANT_MEMORY_H
#define DEVICE_CONSTANT_MEMORY_H

#include "typedefs.hpp"

__device__ __constant__ VG grid_const;
__device__ __constant__ MS setup_const;
__device__ __constant__ int nrays_const;

#endif  // #ifndef DEVICE_CONSTANT_MEMORY_H
