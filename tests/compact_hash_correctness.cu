//
// Created by kr2 on 8/11/21.
//

#include "common.hpp"
#include "cu_common.cuh"
#include "particle.hpp"
#include "pbd.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <random>
#include <set>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

constexpr int NParticles = 4000;

class Random {
 public:
  Random() noexcept : mt(rd()), dist(-border, border){};
  float rand()
  {
    return dist(mt);
  }

 private:
  std::random_device rd{};
  std::mt19937 mt;
  std::uniform_real_distribution<float> dist;
} rd_global;

TEST(COMPACT_HASH, algorithm)
{
  // radius defined in `common.hpp`
#define RAW_PTR(o) (thrust::raw_pointer_cast((o).data()))

  PBDSolver pbd(radius);
  for (int i = 0; i < NParticles; ++i)
    pbd.add_particle(
        SPHParticle(rd_global.rand(), rd_global.rand(), rd_global.rand()));
  pbd.callback();
  const int data_size = (int)pbd.data.size();

  dvector<int> dev_n_neighbor_map(pbd.dev_n_neighbor_map,
                                  pbd.dev_n_neighbor_map + data_size);
  hvector<SPHParticle> &data = pbd.data;
  std::vector<float> bias_data1;
  std::vector<float> bias_data2;

  for (int i = 0; i < data.size(); ++i) {
    hvector<int> cmp_vector_1, cmp_vector_2;

    for (int j = 0; j < data.size(); ++j)
      if (data[i].dist(data[j]) <= radius)
        cmp_vector_1.push_back(j);

    int *hash_map_item = (int *)((char *)pbd.dev_neighbor_map +
                                 i * pbd.pitch_neighbor);
    auto tmp = thrust::device_vector<int>(
        hash_map_item, hash_map_item + dev_n_neighbor_map[i]);
    cmp_vector_2 = hvector<int>(tmp.begin(), tmp.end());

    sort(cmp_vector_1.begin(), cmp_vector_1.end());
    sort(cmp_vector_2.begin(), cmp_vector_2.end());

    EXPECT_EQ(cmp_vector_1.size(), dev_n_neighbor_map[i]);
    EXPECT_TRUE(cmp_vector_1 == cmp_vector_2);
    if (cmp_vector_1 != cmp_vector_2) {
      std::cout << "cur: " << i << std::endl;
      std::cout << "cmp_vector_1: ";
      for (auto &j : cmp_vector_1)
        std::cout << j << " ";
      std::cout << std::endl;

      std::cout << "cmp_vector_2: ";
      for (auto &j : cmp_vector_2)
        std::cout << j << " ";
      std::cout << std::endl;

      if (cmp_vector_1.size() != cmp_vector_2.size()) {
        std::set<int> result1;
        std::set<int> s1(cmp_vector_1.begin(), cmp_vector_1.end()),
            s2(cmp_vector_2.begin(), cmp_vector_2.end());
        std::set_difference(s2.begin(),
                            s2.end(),
                            s1.begin(),
                            s1.end(),
                            std::inserter(result1, result1.end()));
        if (result1.size() != 0) {
          std::cout << "Diff1: ";
          for (auto &k : result1) {
            std::cout << k << ", ";
            std::cout << data[i].dist(data[k]);
            std::cout << std::endl;
            bias_data1.push_back(data[i].dist(data[k]));
          }
        }

        std::set<int> result2;
        std::set_difference(s1.begin(),
                            s1.end(),
                            s2.begin(),
                            s2.end(),
                            std::inserter(result2, result2.end()));
        if (result2.size() != 0) {
          std::cout << "Diff2: ";
          for (auto &k : result2) {
            std::cout << k << ", ";
            std::cout << data[i].dist(data[k]);
            std::cout << std::endl;
            bias_data2.push_back(data[i].dist(data[k]));
          }
        }
      }
    }
  }

  std::cout << "ALL BIAS DATA: ";
  for (auto i : bias_data1) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  for (auto i : bias_data2) {
    std::cout << i << " ";
  }
}
