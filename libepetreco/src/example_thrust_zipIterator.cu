#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>

struct functor
{
  template<typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<3>(t)
          = thrust::get<0>(t) * thrust::get<1>(t) + thrust::get<2>(t);
  }
};

typedef float val_t;
#define N 5

int main()
{
  thrust::device_vector<val_t> A(N);
  thrust::device_vector<val_t> B(N);
  thrust::device_vector<val_t> C(N);
  thrust::device_vector<val_t> D(N);

  for(int id=0; id<N; id++)
  {
    A[id] = id%2+1;
    B[id] = id*id;
    C[id] = N-id;
  }

  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(A.begin(), B.begin(), C.begin(), D.begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(A.end(),   B.end(),   C.end(),   D.end())),
                   functor());

  for(int id=0; id<N; id++)
    std::cout << A[id] << " * " << B[id] << " + " << C[id] << " = " << D[id] << std::endl;

  return 0;
}
