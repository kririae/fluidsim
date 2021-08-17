//
// Created by kr2 on 8/10/21.
//

#include "common.cuh"

[[maybe_unused]] __host__ __device__ float fpow(float a, int b)
{
  float res = 1.0f;
  for (; b; b >>= 1, a *= a)
    if (b & 1)
      res *= a;
  return res;
}

vec3 color_ramp(float t, const color &col_left, const color &col_right)
{
  return (1 - t) * col_left + t * col_right;
}

Random::Random() noexcept : mt(rd()), dist(-border, border)
{
}

float Random::rand()
{
  return dist(mt);
}

[[maybe_unused]] Random rd_global;
