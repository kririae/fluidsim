//
// Created by kr2 on 8/10/21.
//

#include "common.hpp"
#include "cu_common.cuh"
#include <cstdlib>
#include <curand.h>

[[maybe_unused]] CUDA_FUNC_DEC float fast_pow(float a, int b)
{
  float res = 1.0f;
  for (; b; b >>= 1) {
    if (b & 1)
      res *= a;
    a *= a;
  }
  return res;
}

[[maybe_unused]] CUDA_FUNC_DEC vec3 color_ramp(float t,
                                               const color &col_left,
                                               const color &col_right)
{
  return (1 - t) * col_left + t * col_right;
}

Random::Random() noexcept
{
  host_data = (float *)calloc(buffer_size, sizeof(float));
  CUDA_CALL(cudaMalloc((void **)&dev_data, buffer_size * sizeof(float)));
  fill_buffer();
}

void Random::fill_buffer() noexcept
{
  curandGenerator_t gen;
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(curandGenerateUniform(gen, dev_data, buffer_size));
  CUDA_CALL(cudaMemcpy(host_data,
                       dev_data,
                       buffer_size * sizeof(float),
                       cudaMemcpyDeviceToHost));
}

float Random::rand()
{
  float res = host_data[used++];
  if (used == buffer_size) {
    used = 0;
    fill_buffer();
  }
  return res;
}

Random::~Random() noexcept
{
  delete[] host_data;
  cudaFree(dev_data);
}

[[maybe_unused]] Random rd_global;
