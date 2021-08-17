//
// Created by kr2 on 8/11/21.
//

#include "common.hpp"
#include "compact_hash.hpp"
#include "gtest/gtest.h"
#include <random>
#include <vector>

constexpr int NParticles = 8000;

TEST(COMPACT_HASH, algorithm)
{
  float radius = 1.0f;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(-border, border);

  CompactHash ch(radius);
  for (int i = 0; i < NParticles - 200; ++i)
    ch.add_particle(SPHParticle(dist(mt), dist(mt), dist(mt)));
  for (int i = 0; i < 200; ++i)
    ch.add_particle(SPHParticle(border, border, dist(mt)));

  ch.build();
  const auto &data = ch.get_data();

  for (uint i = 0; i < data.size(); ++i) {
    std::vector<int> cmp_vector_1, cmp_vector_2;

    for (uint j = 0; j < data.size(); ++j)
      if (data[i].dist(data[j]) <= radius)
        cmp_vector_1.push_back(j);

    for (uint j = 0; j < ch.n_neighbor(i); ++j)
      cmp_vector_2.push_back(ch.neighbor(i, j));

    sort(cmp_vector_1.begin(), cmp_vector_1.end());
    sort(cmp_vector_2.begin(), cmp_vector_2.end());

    EXPECT_EQ(cmp_vector_1.size(), cmp_vector_2.size());
    EXPECT_TRUE(cmp_vector_1 == cmp_vector_2);
    if (cmp_vector_1 != cmp_vector_2) {
      std::cout << "cmp_vector_1: ";
      for (auto &j : cmp_vector_1)
        std::cout << j << " ";
      std::cout << std::endl;

      std::cout << "cmp_vector_2: ";
      for (auto &j : cmp_vector_2)
        std::cout << j << " ";
      std::cout << std::endl;
    }
  }
}
