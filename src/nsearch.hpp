//
// Created by kr2 on 8/10/21.
//

#ifndef PBF3D_SRC_NSEARCH_HPP_
#define PBF3D_SRC_NSEARCH_HPP_

#include "particle.hpp"

constexpr int MAX_NEIGHBOR_SIZE = 60;

class NSearch {
  // Currently, a poor implementation (for correctness)
 public:
  explicit NSearch(float _radius);
  NSearch(const NSearch &CH) = delete;
  NSearch &operator=(const NSearch &CH) = delete;
  ~NSearch() = default;

  // interface
  void build();
  void add_particle(const SPHParticle &p);
  vector<SPHParticle> &get_data();
  [[nodiscard]] int n_points() const;
  [[nodiscard]] int n_neighbor(uint index) const;
  [[nodiscard]] int neighbor(uint index, uint neighbor_index) const;
  vector<uint> &neighbor_vec(uint index);

 private:
  const float radius, radius2;
  const int n_grids;
  vector<SPHParticle> data{};
  vector<vector<uint>> neighbor_map{};
  vector<vector<int>> hash_map{};

  // hash function
  [[nodiscard]] inline int hash(float x, float y, float z) const;
  [[nodiscard]] inline int hash(const vec3 &p) const;
  [[nodiscard]] inline int hash_from_grid(int u, int v, int w) const;
  [[nodiscard]] inline int hash_from_grid(const ivec3 &p) const;
  [[nodiscard]] inline ivec3 get_grid_index(const vec3 &p) const;
};
#endif  // PBF3D_SRC_NSEARCH_HPP_
