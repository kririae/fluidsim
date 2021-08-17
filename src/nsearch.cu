//
// Created by kr2 on 8/10/21.
//

// currently `index sort`

#include "common.hpp"
#include "nsearch.cuh"
#include "omp.h"
#include <algorithm>

NSearch::NSearch(float _radius)
    : radius(_radius),
      radius2(radius * radius),
      n_grids(int(glm::ceil(2 * border / radius) + 1))
{
  // Initialize to index sort
  hash_map = std::vector<std::vector<int>>(n_grids * n_grids * n_grids);
}

void NSearch::add_particle(const SPHParticle &p)
{
  data.push_back(p);
}

thrust::host_vector<SPHParticle> &NSearch::get_data()
{
  return data;
}

int NSearch::n_points() const
{
  return int(data.size());
}

int NSearch::n_neighbor(uint index) const
{
  return int(neighbor_map[index].size());
}

int NSearch::neighbor(uint index, uint neighbor_index) const
{
  return int(neighbor_map[index][neighbor_index]);
}

void NSearch::build()
{
  neighbor_map = std::vector<std::vector<uint>>(n_points());
  const int data_size = int(data.size());

  // Clear previous information
  std::for_each(hash_map.begin(), hash_map.end(), [&](auto &i) { i.clear(); });

  // Initialize the hash_map
  for (int i = 0; i < data_size; ++i) {
    const int hash_map_index = hash(data[i].pos);
    hash_map[hash_map_index].push_back(i);
  }

#pragma omp parallel for default(none) shared(data_size, n_grids)
  for (int i = 0; i < data_size; ++i) {
    const auto &center = data[i];
    const auto &grid_index = get_grid_index(center.pos);

    for (int u = grid_index.x - 1; u <= grid_index.x + 1; ++u) {
      for (int v = grid_index.y - 1; v <= grid_index.y + 1; ++v) {
        for (int w = grid_index.z - 1; w <= grid_index.z + 1; ++w) {
          if (u < 0 || v < 0 || w < 0 || u >= n_grids || v >= n_grids ||
              w >= n_grids)  // TODO: implement AABB
            continue;

          const auto _hash_index = hash_from_grid(u, v, w);
          const auto &map_item = hash_map[_hash_index];
          std::for_each(map_item.cbegin(), map_item.cend(), [&](int j) {
            if (center.dist2(data[j]) <= radius2 &&
                neighbor_map[i].size() <= ulong(MAX_NEIGHBOR_SIZE))
              neighbor_map[i].push_back(j);
          });
        }
      }
    }
  }
}

inline int NSearch::hash(float x, float y, float z) const
{
  const auto &grid_index = get_grid_index(vec3(x, y, z));
  return hash_from_grid(grid_index);
}

inline int NSearch::hash(const vec3 &p) const
{
  return hash(p.x, p.y, p.z);
}

inline ivec3 NSearch::get_grid_index(const vec3 &p) const
{
  int u = (int)(glm::floor((p.x + border) / radius));
  int v = (int)(glm::floor((p.y + border) / radius));
  int w = (int)(glm::floor((p.z + border) / radius));
  return {u, v, w};
}

inline int NSearch::hash_from_grid(int u, int v, int w) const
{
  return u + v * n_grids + w * n_grids * n_grids;
}

inline int NSearch::hash_from_grid(const ivec3 &p) const
{
  return hash_from_grid(p.x, p.y, p.z);
}

thrust::host_vector<int> &NSearch::neighbor_vec(uint index)
{
  // the result should not be changed
  return neighbor_map[index];
}
