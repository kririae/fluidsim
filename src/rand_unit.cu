//
// Created by kr2 on 8/26/21.
//

#include <thrust/random.h>

#include "rand_unit.cuh"

__device__ RD_GLOBAL::RD_GLOBAL(float _l, float _r) : rand_l(_l), rand_r(_r)
{
}

__device__ float RD_GLOBAL::operator()(unsigned int n)
{
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> dist(rand_l, rand_r);
  rng.discard(n);
  return dist(rng);
}
