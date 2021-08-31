//
// Created by kr2 on 8/10/21.
//

#include "cu_common.cuh"
#include "gui.hpp"
#include "particle.hpp"
#include "pbd.hpp"
#include "rand_unit.cuh"
#include <chrono>
#include <iostream>

#define RAW_PTR(o) (thrust::raw_pointer_cast((o).data()))

constexpr float denom_epsilon = 20.0f;

PBDSolver::PBDSolver(float _radius)
    : radius(_radius), radius2(_radius * _radius)
{
  mass = 4.0f / 3.0f * glm::pi<float>() * radius2;
  n_grids = int(glm::ceil(2 * border / radius) + 1);
}

// void PBDSolver::set_gui(RTGUI_particles *gui) noexcept
// {
//   gui_ptr = gui;
// }

void PBDSolver::update_gui(RTGUI_particles *gui_ptr) noexcept
{
#pragma unroll
  for (int i = 0; i < n_substeps; ++i)
    this->substep();
  gui_ptr->set_particles(data);
}

void PBDSolver::substep()
{
  // data -> [__CUDA_OPERATIONS__] -> data (at all)
  // For compatibility consideration, cannot use modern CPP
  static int interval = 1;

  // for data_size linear parallel
  const int data_size = int(data.size());

  constexpr int threads_per_block = 1024;
  const int num_blocks = (data_size + threads_per_block - 1) /
                         threads_per_block;

  int *dev_neighbor_map;
  size_t pitch_neighbor{0};
  dvector<float> lambda(data_size), c_i(data_size);
  dvector<SPHParticle> pre_data(data.begin(), data.end());
  dvector<SPHParticle> dev_data = pre_data;
  dvector<int> dev_n_neighbor_map(data_size, 0);

  // Apply forces, on host
  apply_force<<<num_blocks, threads_per_block>>>(
      RAW_PTR(dev_data), data_size, ext_f, delta_t);

  cudaDeviceSynchronize();
  // find all neighbors
  auto dt_start = std::chrono::system_clock::now();

  // Block of building
  {
    const auto map_size = (size_t)glm::pow(n_grids, 3.0f);
    dvector<int> hash_map_mutex(map_size, 0);
    dvector<int> dev_n_hash_map(map_size, 0);

    int *dev_hash_map;
    size_t pitch_hash;
    CUDA_CALL(
        cudaMallocPitch(&dev_hash_map,
                        &pitch_hash,
                        MAX_NEIGHBOR_SIZE * sizeof(int),  // risk of exceed
                        map_size));
    CUDA_CALL(cudaMallocPitch(&dev_neighbor_map,
                              &pitch_neighbor,
                              MAX_NEIGHBOR_SIZE * sizeof(int),
                              data_size));

    build_hash_map<<<num_blocks, threads_per_block>>>(RAW_PTR(dev_data),
                                                      data_size,
                                                      dev_hash_map,
                                                      RAW_PTR(dev_n_hash_map),
                                                      n_grids,
                                                      pitch_hash,
                                                      RAW_PTR(hash_map_mutex));

    build_neighbor_map<<<num_blocks, threads_per_block>>>(
        RAW_PTR(dev_data),
        data_size,
        dev_neighbor_map,
        RAW_PTR(dev_n_neighbor_map),
        dev_hash_map,
        thrust::raw_pointer_cast(dev_n_hash_map.data()),
        n_grids,
        pitch_neighbor,
        pitch_hash);

    cudaDeviceSynchronize();
    CUDA_CALL(cudaFree(dev_hash_map));
  }

  auto dt_end = std::chrono::system_clock::now();

  // Jacobi iteration
  auto start = std::chrono::system_clock::now();
  int iter_cnt = iter;
  while (iter_cnt--) {
    fill_lambda<<<num_blocks, threads_per_block>>>(RAW_PTR(dev_data),
                                                   data_size,
                                                   dev_neighbor_map,
                                                   RAW_PTR(dev_n_neighbor_map),
                                                   pitch_neighbor,
                                                   RAW_PTR(c_i),
                                                   RAW_PTR(lambda),
                                                   rho_0,
                                                   mass);
    apply_motion<<<num_blocks, threads_per_block>>>(RAW_PTR(dev_data),
                                                    data_size,
                                                    RAW_PTR(dev_n_neighbor_map),
                                                    dev_neighbor_map,
                                                    RAW_PTR(lambda),
                                                    RAW_PTR(c_i),
                                                    pitch_neighbor,
                                                    rho_0);
  }

  // update all velocity
  update_velocity<<<num_blocks, threads_per_block>>>(
      RAW_PTR(dev_data), RAW_PTR(pre_data), data_size, delta_t);

  cudaDeviceSynchronize();

  // Copy device vector back to host
  CUDA_CALL(cudaMemcpy(&data[0],
                       RAW_PTR(dev_data),
                       sizeof(SPHParticle) * data_size,
                       cudaMemcpyDeviceToHost));

  CUDA_CALL(cudaFree(dev_neighbor_map));

  // Logging part
  if ((--interval) != 0)
    return;

  interval = 60;
  std::cout << "--- substep start (interval: 60) ---" << std::endl;
  std::cout << "NParticles: " << data_size << std::endl;
  std::chrono::duration<float> dt_diff = dt_end - dt_start;
  std::cout << "data_structure building complete: " << dt_diff.count() * 1000
            << "ms" << std::endl;
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<float> diff = end - start;
  std::cout << "calculation complete: " << diff.count() * 1000 << "ms"
            << std::endl;
  std::cout << "avg c_i: "
            << thrust::reduce(
                   c_i.begin(), c_i.end(), 0.0f, thrust::plus<float>()) /
                   static_cast<double>(data_size)
            << " | n_neighbor: "
            << thrust::reduce(dev_n_neighbor_map.begin(),
                              dev_n_neighbor_map.end(),
                              0,
                              thrust::plus<int>()) /
                   data_size
            << std::endl;
  std::cout << std::endl;
}

void PBDSolver::add_particle(const SPHParticle &p)
{
  data.push_back(p);
}

const hvector<SPHParticle> &PBDSolver::get_data()
{
  return data;
}

__device__ void PBDSolver::constraint_to_border(SPHParticle &p)
{
  // TODO: the same offset on two dimensions
  static unsigned int a = 0;
  auto _rd = RD_GLOBAL(-1.0f, 1.0f);
  atomicAdd(&a, 1);
  p.pos += epsilon * vec3(_rd(a), 0.0f, 0.0f);
  atomicAdd(&a, 1);
  p.pos += epsilon * vec3(0.0f, _rd(a), 0.0f);
  atomicAdd(&a, 1);
  p.pos += epsilon * vec3(0.0f, 0.0f, _rd(a));
  p.pos.x = glm::clamp(p.pos.x, -border, border);
  p.pos.y = glm::clamp(p.pos.y, -border, border);
  p.pos.z = glm::clamp(p.pos.z, -border, border);
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
  float delta_q = 0.5f * h;
  float r = glm::length(p_i.pos - p_j.pos);
  return -k * fast_pow(poly6(r, h) / poly6(delta_q, h), n);
}

__global__ void update_velocity(SPHParticle *dev_data,
                                SPHParticle *pre_data,
                                int data_size,
                                float delta_t)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    auto &p = dev_data[i];
    PBDSolver::constraint_to_border(p);
    p.v = 1.0f / delta_t * (p.pos - pre_data[i].pos);
  }
}

__global__ void apply_force(SPHParticle *dev_data,
                            int data_size,
                            vec3 ext_f,
                            float delta_t)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    dev_data[i].v += delta_t * ext_f;
    dev_data[i].pos += delta_t * dev_data[i].v;
    PBDSolver::constraint_to_border(dev_data[i]);
  }
}

__global__ void fill_lambda(SPHParticle *dev_data,
                            size_t data_size,
                            int *dev_neighbor_map,
                            const int *dev_n_neighbor_map,
                            size_t pitch_neighbor,
                            float *c_i,
                            float *lambda,
                            float rho_0,
                            float mass)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    // --- calculate rho ---
    float rho = 0;
    int *dev_neighbor_vec = (int *)((char *)dev_neighbor_map +
                                    i * pitch_neighbor);
    for (int index = 0; index < dev_n_neighbor_map[i]; ++index) {
      int j = dev_neighbor_vec[index];
      rho += mass * PBDSolver::poly6(
                        glm::length(dev_data[i].pos - dev_data[j].pos), radius);
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
              dev_data[i].pos - dev_data[neighbor_index].pos, radius);
        }
      } else {
        grad_c = -PBDSolver::grad_spiky(dev_data[i].pos - dev_data[j].pos,
                                        radius);
      }

      grad_c = 1.0f / rho_0 * grad_c;
      _denom += fast_pow(glm::length(grad_c), 2.0f);
    }

    lambda[i] = -c_i[i] / (_denom + denom_epsilon);
  }
}

__global__ void apply_motion(SPHParticle *dev_data,
                             size_t data_size,
                             const int *dev_n_neighbor_map,
                             const int *dev_neighbor_map,
                             const float *lambda,
                             const float *c_i,
                             size_t pitch,
                             float rho_0)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    vec3 delta_p_i(0.0f);
    int *dev_neighbor_vec = (int *)((char *)dev_neighbor_map + i * pitch);
    for (int index = 0; index < dev_n_neighbor_map[i]; ++index) {
      int j = dev_neighbor_vec[index];
      delta_p_i +=
          (lambda[i] + lambda[j] +
           PBDSolver::compute_s_corr(dev_data[i], dev_data[j], radius)) *
          PBDSolver::grad_spiky(dev_data[i].pos - dev_data[j].pos, radius);
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
                               size_t pitch_hash,
                               int *hash_map_mutex)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    int hash_map_index = PBDSolver::hash(dev_data[i].pos, n_grids);
    int *dev_hash_map_item = (int *)((char *)dev_hash_map +
                                     pitch_hash * hash_map_index);
    while (atomicCAS(&hash_map_mutex[hash_map_index], 0, 1) != 0)
      ;
    if (dev_n_hash_map[hash_map_index] < PBDSolver::MAX_NEIGHBOR_SIZE) {
      atomicExch((int *)&(dev_hash_map_item[dev_n_hash_map[hash_map_index]]),
                 (int)i);
      atomicAdd((int *)&(dev_n_hash_map[hash_map_index]), 1);
    }
    atomicExch(&hash_map_mutex[hash_map_index], 0);
  }
}

__global__ void build_neighbor_map(SPHParticle *dev_data,
                                   int data_size,
                                   int *dev_neighbor_map,
                                   int *dev_n_neighbor_map,
                                   const int *dev_hash_map,
                                   const int *dev_n_hash_map,
                                   int n_grids,
                                   size_t pitch_neighbor,
                                   size_t pitch_hash)
{
  unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < data_size) {
    const auto &center = dev_data[i];
    const auto grid_index = PBDSolver::get_grid_index(center.pos);
    int *dev_neighbor_map_item = (int *)((char *)dev_neighbor_map +
                                         pitch_neighbor * i);

    for (int u = grid_index.x - 1; u <= grid_index.x + 1; ++u) {
      for (int v = grid_index.y - 1; v <= grid_index.y + 1; ++v) {
        for (int w = grid_index.z - 1; w <= grid_index.z + 1; ++w) {
          if (u < 0 || v < 0 || w < 0 || u > n_grids || v > n_grids ||
              w > n_grids)
            continue;

          const int hash_map_index = PBDSolver::hash_from_grid(
              u, v, w, n_grids);
          int *dev_hash_map_item = (int *)((char *)dev_hash_map +
                                           pitch_hash * hash_map_index);
          for (int k = 0; k < dev_n_hash_map[hash_map_index]; ++k) {
            int j = dev_hash_map_item[k];
            if (center.dist2(dev_data[j]) <= radius2 &&
                dev_n_neighbor_map[i] < PBDSolver::MAX_NEIGHBOR_SIZE) {
              // not required
              atomicExch((int *)&(dev_neighbor_map_item[dev_n_neighbor_map[i]]),
                         j);
              atomicAdd(&(dev_n_neighbor_map[i]), 1);
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
