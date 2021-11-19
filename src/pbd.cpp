//
// Created by kr2 on 9/24/21.
//

// CPU version of pbd.cu,
// for my new computer cannot run pbd.cu

/*
#include "pbd.hpp"
#include "common.hpp"
#include "gui.hpp"
#include "particle.hpp"
#include <chrono>
#include <iostream>

constexpr float denom_epsilon = 100.0f;

PBDSolver::PBDSolver(float _radius)
    : radius(_radius), radius2(_radius * _radius)
{
  // mass = 4.0f / 3.0f * glm::pi<float>() * radius2;
  mass = 1.0f;
  n_grids = int(glm::ceil(2 * border / radius) + 1);
}

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

  constexpr int threads_per_block = 2;
  const int num_blocks = (data_size + threads_per_block - 1) /
      threads_per_block;

  int *dev_neighbor_map;
  size_t pitch_neighbor{0};
  std::vector<float> lambda(data_size), c_i(data_size);
  std::vector<SPHParticle> pre_data(data.begin(), data.end());
  std::vector<SPHParticle> dev_data = pre_data;
  std::vector<int> dev_n_neighbor_map(data_size, 0);

  // Apply forces, on host
  apply_force(dev_data.data(), data_size, ext_f, delta_t);

  // find all neighbors
  auto dt_start = std::chrono::system_clock::now();

  // Block of building
  {
    const auto map_size = (size_t)glm::pow(n_grids, 3.0f);
    std::vector<int> hash_map_mutex(map_size, 0);
    std::vector<int> dev_n_hash_map(map_size, 0);

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

// __host__
void PBDSolver::constraint_to_border(SPHParticle &p)
{
  p.pos.x = glm::clamp(p.pos.x, -border, border);
  p.pos.y = glm::clamp(p.pos.y, -border, border);
  p.pos.z = glm::clamp(p.pos.z, -border, border);
}

// On both platform
float PBDSolver::poly6(float r, float h) noexcept
{
  r = glm::clamp(glm::abs(r), 0.0f, h);
  const float t = (h * h - r * r) / (h * h * h);
  return 315.0f / (64 * glm::pi<float>()) * t * t * t;
}

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

void update_velocity(SPHParticle *dev_data,
                                SPHParticle *pre_data,
                                int data_size,
                                float delta_t)
{
  for (int i = 0; i < data_size; ++i) {
    auto &p = dev_data[i];
    PBDSolver::constraint_to_border(p);
    p.v = 1.0f / delta_t * (p.pos - pre_data[i].pos);
  }
}

void apply_force(SPHParticle *dev_data,
                            int data_size,
                            vec3 ext_f,
                            float delta_t)
{
  for (int i = 0; i < data_size; ++i) {
    dev_data[i].v += delta_t * ext_f;
    dev_data[i].pos += delta_t * dev_data[i].v;
    PBDSolver::constraint_to_border(dev_data[i]);
  }
}

void fill_lambda(SPHParticle *dev_data,
                            size_t data_size,
                            int *dev_neighbor_map,
                            const int *dev_n_neighbor_map,
                            size_t pitch_neighbor,
                            float *c_i,
                            float *lambda,
                            float rho_0,
                            float mass)
{
  for (int i = 0; i < data_size; ++i) {
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

void apply_motion(SPHParticle *dev_data,
                             size_t data_size,
                             const int *dev_n_neighbor_map,
                             const int *dev_neighbor_map,
                             const float *lambda,
                             const float *c_i,
                             size_t pitch,
                             float rho_0)
{
  for (int i = 0; i < data_size; ++i) {
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

*/