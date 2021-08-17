//
// Created by kr2 on 8/11/21.
//

#include "common.hpp"
#include "pbd.cuh"
#include "gtest/gtest.h"
#include <random>

static void print(vec3 v)
{
  std::cout << v.x << " " << v.y << " " << v.z << std::endl;
}

[[maybe_unused]] [[maybe_unused]] Random rd_global;

TEST(PBD_SOLVER, grad_c)
{
  PBDSolver pbd(1.2f);
  pbd.add_particle(SPHParticle(0, 1, 0));
  pbd.add_particle(SPHParticle(0, 2, 0));

  pbd.ch_ptr->build();
  print(pbd.grad_c(0, 0, 1));
}