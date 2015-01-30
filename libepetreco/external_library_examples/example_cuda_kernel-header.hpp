/* 
 * File:   example_cuda_kernel-header.hpp
 * Author: malte
 *
 * Created on 23. Januar 2015, 12:15
 */

#ifndef EXAMPLE_CUDA_KERNEL_HEADER_HPP
#define	EXAMPLE_CUDA_KERNEL_HEADER_HPP

#define TPB 256   // threads per block

template< typename T >
__global__
void add(
      T * const a_devi, T * const b_devi, T * const result_devi,
      int const n_devi );



#include "example_cuda_kernel-header.tpp"

#endif	/* EXAMPLE_CUDA_KERNEL_HEADER_HPP */
