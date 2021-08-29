//
// Created by kr2 on 8/24/21.
//

#include <cstdio>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
template<typename T> using dvector = thrust::device_vector<T>;

#define CUDA_CALL(x) \
  do { \
    if ((x) != cudaSuccess) { \
      printf("Error at %s:%d\n", __FILE__, __LINE__); \
    } \
  } while (0)