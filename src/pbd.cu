//
// Created by kr2 on 8/10/21.
//

#include "cu_common.cuh"
#include "particle.hpp"
#include "pbd.hpp"
#include <chrono>
#include <iostream>
#include <thrust/random.h>

constexpr float denom_epsilon = 20.0f;

static CUDA_FUNC_DEC float dev_rand()
{
  // TODO
  thrust::random::default_random_engine rng;
  thrust::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  return dist(rng);
}

PBDSolver::PBDSolver(float _radius)
    : radius(_radius), radius2(_radius * _radius)
{
  mass = 4.0f / 3.0f * glm::pi<float>() * radius2;
  n_grids = int(glm::ceil(2 * border / radius) + 1);
}

void PBDSolver::set_gui(RTGUI_particles *gui) noexcept
{
  gui_ptr = gui;
}

void PBDSolver::callback()
{
  assert(gui_ptr != nullptr);

  static int interval = 60;

  constexpr int threads_per_block = 1024;
  const int data_size = int(data.size());
  const int num_blocks = (data_size + threads_per_block) / threads_per_block;

  CUDA_CALL(cudaMalloc(&_lambda, sizeof(float) * data_size));
  CUDA_CALL(cudaMalloc(&c_i, sizeof(float) * data_size));

  SPHParticle *pre_data, *dev_data;
  CUDA_CALL(cudaMalloc(&pre_data, sizeof(SPHParticle) * data_size));
  CUDA_CALL(cudaMalloc(&dev_data, sizeof(SPHParticle) * data_size));

  // 1 H to D memory copy.
  // Copy particle data to device
  CUDA_CALL(cudaMemcpy(pre_data,
                       data.data(),
                       data_size * sizeof(SPHParticle),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dev_data,
                       pre_data,
                       data_size * sizeof(SPHParticle),
                       cudaMemcpyDeviceToDevice));

  // Apply forces, on host
  apply_force<<<num_blocks, threads_per_block>>>(
      dev_data, delta_t, ext_f, data_size, border);

  // find all neighbors
  auto dt_start = std::chrono::system_clock::now();

  // Block of building
  {
    int *dev_hash_map, *dev_n_hash_map;

    size_t pitch_hash;
    CUDA_CALL(
        cudaMallocPitch(&dev_hash_map,
                        &pitch_hash,
                        MAX_NEIGHBOR_SIZE * sizeof(int),  // risk of exceed
                        n_grids * n_grids * n_grids));
    CUDA_CALL(
        cudaMalloc(&dev_n_hash_map, sizeof(int) * n_grids * n_grids * n_grids));
    CUDA_CALL(cudaMemset(
        dev_n_hash_map, 0, sizeof(int) * n_grids * n_grids * n_grids));

    CUDA_CALL(cudaMallocPitch(
        &dev_neighbor_map, &pitch, MAX_NEIGHBOR_SIZE * sizeof(int), data_size));
    CUDA_CALL(cudaMalloc(&dev_n_neighbor_map, sizeof(int) * data_size));
    CUDA_CALL(cudaMemset(dev_n_neighbor_map, 0, sizeof(int) * data_size));

    build_hash_map<<<num_blocks, threads_per_block>>>(
        dev_data, data_size, dev_hash_map, dev_n_hash_map, n_grids, pitch_hash);

    build_neighbor_map<<<num_blocks, threads_per_block>>>(dev_data,
                                                          data_size,
                                                          dev_neighbor_map,
                                                          dev_n_neighbor_map,
                                                          dev_hash_map,
                                                          dev_n_hash_map,
                                                          n_grids,
                                                          pitch,
                                                          pitch_hash);

    CUDA_CALL(cudaFree(dev_hash_map));
    CUDA_CALL(cudaFree(dev_n_hash_map));
  }

  auto dt_end = std::chrono::system_clock::now();

  auto start = std::chrono::system_clock::now();
  // Jacobi iteration
  int iter_cnt = iter;
  while (iter_cnt--) {
    fill_lambda<<<num_blocks, threads_per_block>>>(dev_data,
                                                   dev_neighbor_map,
                                                   pitch,
                                                   dev_n_neighbor_map,
                                                   _lambda,
                                                   c_i,
                                                   data_size,
                                                   rho_0,
                                                   radius,
                                                   mass);
    apply_motion<<<num_blocks, threads_per_block>>>(dev_data,
                                                    dev_neighbor_map,
                                                    pitch,
                                                    dev_n_neighbor_map,
                                                    _lambda,
                                                    c_i,
                                                    data_size,
                                                    rho_0,
                                                    radius);
  }

  float c_i_sum = 0;
  // update all velocity
  update_velocity<<<num_blocks, threads_per_block>>>(
      dev_data, pre_data, delta_t, data_size, border);

  // Copy device vector back to host
  CUDA_CALL(cudaMemcpy(&data[0],
                       dev_data,
                       sizeof(SPHParticle) * data_size,
                       cudaMemcpyDeviceToHost));

  gui_ptr->set_particles(data);

  CUDA_CALL(cudaFree(_lambda));
  CUDA_CALL(cudaFree(c_i));
  CUDA_CALL(cudaFree(pre_data));
  CUDA_CALL(cudaFree(dev_data));
  CUDA_CALL(cudaFree(dev_neighbor_map));
  CUDA_CALL(cudaFree(dev_n_neighbor_map));

  // Logging part
  if ((--interval) != 0)
    return;

  interval = 60;
  std::cout << "--- callback start (interval: 60) ---" << std::endl;
  std::cout << "NParticles: " << data_size << std::endl;
  std::chrono::duration<float> dt_diff = dt_end - dt_start;
  std::cout << "data_structure building complete: " << dt_diff.count() * 1000
            << "ms" << std::endl;
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<float> diff = end - start;
  std::cout << "calculation complete: " << diff.count() * 1000 << "ms"
            << std::endl;
  std::cout << "avg c_i: <" << c_i_sum / data_size << "> | n_neighbor: <null>"
            << std::endl;
  std::cout << std::endl;
}

void PBDSolver::add_particle(const SPHParticle &p)
{
  data.push_back(p);
}

hvector<SPHParticle> &PBDSolver::get_data()
{
  return data;
}

__device__ void PBDSolver::constraint_to_border(SPHParticle &p, float _border)
{
  p.pos += epsilon * vec3(dev_rand(), dev_rand(), dev_rand());
  p.pos.x = glm::clamp(p.pos.x, -_border, _border);
  p.pos.y = glm::clamp(p.pos.y, -_border, _border);
  p.pos.z = glm::clamp(p.pos.z, -_border, _border);
}

// On both platform
CUDA_FUNC_DEC
float PBDSolver::poly6(float r, float h) noexcept
{
  r = glm::clamp(glm::abs(r), 0.0f, h);
  const float t = (h * h - r * r) / (h * h * h);
  return 315.0f / (64 * glm::pi<float>()) * t * t * t;
}

CUDA_FUNC_DEC
vec3 PBDSolver::grad_spiky(vec3 v, float h) noexcept
{
  float len = glm::length(v);
  vec3 res(0.0f);
  if (0 < len && len <= h)
    res = float(-45 / (glm::pi<float>() * fast_pow(h, 6)) *
                fast_pow(h - len, 2)) *
          v / len;
  return res;
}

CUDA_FUNC_DEC
float PBDSolver::compute_s_corr(const SPHParticle &p_i,
                                const SPHParticle &p_j,
                                float h)
{
  float k = 0.1f;  // k
  float n = 4.0f;
  float delta_q = 0.3f * h;
  float r = glm::length(p_i.pos - p_j.pos);
  return -k * fast_pow(poly6(r, h) / poly6(delta_q, h), n);
}

__global__ void update_velocity(SPHParticle *dev_data,
                                SPHParticle *pre_data,
                                float delta_t,
                                int data_size,
                                float _border)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    auto &p = dev_data[i];
    PBDSolver::constraint_to_border(p, _border);
    p.v = 1.0f / delta_t * (p.pos - pre_data[i].pos);
  }
}

__global__ void apply_force(SPHParticle *dev_data,
                            float delta_t,
                            vec3 ext_f,
                            int data_size,
                            float _border)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    dev_data[i].v += delta_t * ext_f;
    dev_data[i].pos += delta_t * dev_data[i].v;
    PBDSolver::constraint_to_border(dev_data[i], _border);
  }
}

__global__ void fill_lambda(SPHParticle *dev_data,
                            int *dev_neighbor_map,
                            size_t pitch,
                            const int *dev_n_neighbor_map,
                            float *_lambda,
                            float *c_i,
                            size_t data_size,
                            float rho_0,
                            float _radius,
                            float mass)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    // --- calculate rho ---
    float rho = 0;
    int *dev_neighbor_vec = (int *)((char *)dev_neighbor_map + i * pitch);
    for (int index = 0; index < dev_n_neighbor_map[i]; ++index) {
      int j = dev_neighbor_vec[index];
      rho += mass *
             PBDSolver::poly6(glm::length(dev_data[i].pos - dev_data[j].pos),
                              _radius);
    }

    c_i[i] = glm::max(rho / rho_0 - 1, 0.0f);

    // --- calculate grad_c ---
    float _denom = 0.0f;
    for (int index = 0; index < dev_n_neighbor_map[i]; ++index) {
      int j = dev_neighbor_vec[index];
      vec3 grad_c(0.0f);
      if (i == j) {
        for (int k = 0; k < dev_n_neighbor_map[i]; ++k) {
          const int neighbor_index = dev_neighbor_vec[k];
          grad_c += PBDSolver::grad_spiky(
              dev_data[i].pos - dev_data[neighbor_index].pos, _radius);
        }
      } else {
        grad_c = -PBDSolver::grad_spiky(dev_data[i].pos - dev_data[j].pos,
                                        _radius);
      }

      grad_c = 1.0f / rho_0 * grad_c;
      _denom += fast_pow(glm::length(grad_c), 2.0f);
    }

    _lambda[i] = -c_i[i] / (_denom + denom_epsilon);
  }
}

__global__ void apply_motion(SPHParticle *dev_data,
                             const int *dev_neighbor_map,
                             size_t pitch,
                             const int *dev_n_neighbor_map,
                             const float *_lambda,
                             const float *c_i,
                             size_t data_size,
                             float rho_0,
                             float _radius)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < data_size) {
    vec3 delta_p_i(0.0f);
    int *dev_neighbor_vec = (int *)((char *)dev_neighbor_map + i * pitch);
    for (int index = 0; index < dev_n_neighbor_map[i]; ++index) {
      int j = dev_neighbor_vec[index];
      assert(j != -1);
      delta_p_i +=
          (_lambda[i] + _lambda[j] +
           PBDSolver::compute_s_corr(dev_data[i], dev_data[j], _radius)) *
          PBDSolver::grad_spiky(dev_data[i].pos - dev_data[j].pos, _radius);
    }

    delta_p_i *= 1.0f / rho_0;
    dev_data[i].pos += delta_p_i;
    dev_data[i].rho = glm::clamp(c_i[i], 0.0f, 1.0f);
  }
}

// Data struct implementation
__global__ void build_hash_map(SPHParticle *dev_data,
                               int data_size,
                               int *dev_hash_map,
                               int *dev_n_hash_map,
                               int n_grids,
                               size_t pitch_hash)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    const int hash_map_index = PBDSolver::hash(dev_data[i].pos, n_grids);
    int *dev_hash_map_item = (int *)((char *)dev_hash_map +
                                     pitch_hash * hash_map_index);
    if (dev_n_hash_map[hash_map_index] < PBDSolver::MAX_NEIGHBOR_SIZE) {
      atomicAdd(&dev_n_hash_map[hash_map_index], 1);
      dev_hash_map_item[dev_n_hash_map[hash_map_index]] = (int)i;
    } else {
      // assert(false);
    }
  }
}

__global__ void build_neighbor_map(SPHParticle *dev_data,
                                   int data_size,
                                   int *dev_neighbor_map,
                                   int *dev_n_neighbor_map,
                                   const int *dev_hash_map,
                                   const int *dev_n_hash_map,
                                   int n_grids,
                                   size_t pitch,
                                   size_t pitch_hash)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    const auto &center = dev_data[i];
    const auto grid_index = PBDSolver::get_grid_index(center.pos);
    int *dev_neighbor_map_item = (int *)((char *)dev_neighbor_map + pitch * i);

    for (int u = grid_index.x - 1; u <= grid_index.x + 1; ++u) {
      for (int v = grid_index.y - 1; v <= grid_index.y + 1; ++v) {
        for (int w = grid_index.z - 1; w <= grid_index.z + 1; ++w) {
          if (u < 0 || v < 0 || w < 0 || u >= n_grids || v >= n_grids ||
              w >= n_grids)
            continue;

          const int hash_map_index = PBDSolver::hash_from_grid(
              u, v, w, n_grids);
          int *dev_hash_map_item = (int *)((char *)dev_hash_map +
                                           pitch_hash * hash_map_index);
          for (int k = 0; k < dev_n_hash_map[hash_map_index]; ++k) {
            int j = dev_hash_map_item[k];
            if (center.dist(dev_data[j]) <= radius &&
                dev_n_neighbor_map[i] < PBDSolver::MAX_NEIGHBOR_SIZE) {
              atomicAdd(&dev_n_neighbor_map[i], 1);
              dev_neighbor_map_item[dev_n_neighbor_map[i]] = j;
            }
          }
        }
      }
    }
  }
}

CUDA_FUNC_DEC
inline int PBDSolver::hash(float x, float y, float z, int n_grids)
{
  const auto &grid_index = get_grid_index(vec3(x, y, z));
  return hash_from_grid(grid_index, n_grids);
}

CUDA_FUNC_DEC
inline int PBDSolver::hash(const vec3 &p, int n_grids)
{
  return hash(p.x, p.y, p.z, n_grids);
}

CUDA_FUNC_DEC
inline ivec3 PBDSolver::get_grid_index(const vec3 &p)
{
  int u = (int)(glm::floor((p.x + border) / ::radius));
  int v = (int)(glm::floor((p.y + border) / ::radius));
  int w = (int)(glm::floor((p.z + border) / ::radius));
  return {u, v, w};
}

CUDA_FUNC_DEC
inline int PBDSolver::hash_from_grid(int u, int v, int w, int n_grids)
{
  return u + v * n_grids + w * n_grids * n_grids;
}

CUDA_FUNC_DEC
inline int PBDSolver::hash_from_grid(const ivec3 &p, int n_grids)
{
  return hash_from_grid(p.x, p.y, p.z, n_grids);
}
