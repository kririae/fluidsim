//
// Created by kr2 on 8/10/21.
//

#include "common.hpp"
#include "cu_common.cuh"

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
