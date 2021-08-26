//
// Created by kr2 on 8/26/21.
//

#ifndef PBF3D_SRC_RAND_UNIT_CUH_
#define PBF3D_SRC_RAND_UNIT_CUH_

#include "common.hpp"
#include <thrust/random.h>

struct RD_GLOBAL {
  float rand_l, rand_r;

  __attribute__((device)) RD_GLOBAL(float _l, float _r);
  __attribute__((device)) float operator()(unsigned int n);
};

#endif  // PBF3D_SRC_RAND_UNIT_CUH_
