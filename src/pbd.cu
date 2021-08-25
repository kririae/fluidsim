//
// Created by kr2 on 8/10/21.
//

#include "cu_common.cuh"
#include "pbd.hpp"
#include <chrono>
#include <iostream>

constexpr float denom_epsilon = 20.0f;

PBDSolver::PBDSolver(float _radius)
    : ch_ptr(std::make_shared<NSearch>(_radius)),
      radius(_radius),
      radius2(_radius * _radius)
{
  mass = 4.0f / 3.0f * glm::pi<float>() * radius2;
}

void PBDSolver::set_gui(RTGUI_particles *gui) noexcept
{
  gui_ptr = gui;
}

void PBDSolver::callback()
{
  assert(gui_ptr != nullptr);

  static int interval = 60;

  hvector<SPHParticle> &host_data = get_data();
  const hvector<SPHParticle> pre_data = host_data;  // copy
  const int data_size = int(host_data.size());

  // Apply forces, on host
  std::for_each(host_data.begin(), host_data.end(), [&](SPHParticle &item) {
    item.v += delta_t * ext_f;
    item.pos += delta_t * item.v;
    constraint_to_border(item);
  });

  // find all neighbors
  auto dt_start = std::chrono::system_clock::now();
  ch_ptr->build();
  auto dt_end = std::chrono::system_clock::now();

  auto start = std::chrono::system_clock::now();

  // Jacobi iteration
  SPHParticle *dev_data;
  CUDA_CALL(cudaMalloc(&dev_data, data_size * sizeof(SPHParticle)));
  CUDA_CALL(cudaMemcpy(dev_data,
                       host_data.data(),
                       data_size * sizeof(SPHParticle),
                       cudaMemcpyHostToDevice));

  int *dev_neighbor_map;  // data_size * MAX_NEIGHBOR_SIZE
  int *dev_n_neighbor_map;
  size_t pitch;

  CUDA_CALL(cudaMallocPitch(&dev_neighbor_map,
                            &pitch,
                            ch_ptr->MAX_NEIGHBOR_SIZE * sizeof(int),
                            data_size));
  CUDA_CALL(cudaMalloc(&dev_n_neighbor_map, sizeof(int) * data_size));
  auto *n_neighbor_map = (int *)malloc(sizeof(int) * data_size);
  for (int i = 0; i < data_size; ++i)
    n_neighbor_map[i] = ch_ptr->n_neighbor(i);
  CUDA_CALL(cudaMemcpy(dev_n_neighbor_map,
                       n_neighbor_map,
                       data_size * sizeof(int),
                       cudaMemcpyHostToDevice));
  free(n_neighbor_map);

  // Copy nsearch data onto GPU
  for (int i = 0; i < data_size; ++i) {
    int *dev_neighbor_vec = (int *)((char *)dev_neighbor_map + i * pitch);
    const auto &neighbor_vec = ch_ptr->neighbor_vec(i);
    CUDA_CALL(cudaMemcpy(dev_neighbor_vec,
                         neighbor_vec.data(),
                         neighbor_vec.size() * sizeof(int),
                         cudaMemcpyHostToDevice));
  }

  float *_lambda, *c_i;
  CUDA_CALL(cudaMalloc(&_lambda, sizeof(float) * data_size));
  CUDA_CALL(cudaMalloc(&c_i, sizeof(float) * data_size));

  const int threads_per_block = 128;
  const int num_blocks = (data_size + threads_per_block - 1) /
                         threads_per_block;

  auto end = std::chrono::system_clock::now();
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

  // Copy device vector back to host
  CUDA_CALL(cudaMemcpy(&host_data[0],
                       dev_data,
                       sizeof(SPHParticle) * data_size,
                       cudaMemcpyDeviceToHost));
  // float c_i_sum = thrust::reduce(
  //     c_i, c_i + data_size, 0.0f, thrust::plus<float>());
  float c_i_sum = 0;
  CUDA_CALL(cudaFree(dev_neighbor_map));
  CUDA_CALL(cudaFree(dev_data));
  CUDA_CALL(cudaFree(_lambda));
  CUDA_CALL(cudaFree(c_i));

  // update all velocity
  for (int i = 0; i < data_size; ++i) {
    auto &p = host_data[i];
    constraint_to_border(p);
    p.v = 1.0f / delta_t * (p.pos - pre_data[i].pos);
  }

  gui_ptr->set_particles(host_data);

  // Logging part
  if ((--interval) != 0)
    return;

  interval = 60;
  std::cout << "--- callback start (interval: 60) ---" << std::endl;
  std::cout << "NParticles: " << data_size << std::endl;
  std::chrono::duration<float> dt_diff = dt_end - dt_start;
  std::cout << "data_structure building complete: " << dt_diff.count() * 1000
            << "ms" << std::endl;
  std::chrono::duration<float> diff = end - start;
  std::cout << "calculation complete: " << diff.count() * 1000 << "ms"
            << std::endl;
  std::cout << "avg c_i: " << c_i_sum / data_size << " | n_neighbor: <null>"
            << std::endl;
  std::cout << std::endl;
}

void PBDSolver::add_particle(const SPHParticle &p)
{
  assert(ch_ptr != nullptr);
  ch_ptr->add_particle(p);
}

hvector<SPHParticle> &PBDSolver::get_data()
{
  return ch_ptr->get_data();
}

void PBDSolver::constraint_to_border(SPHParticle &p)
{
  p.pos += epsilon * vec3(rd_global.rand(), rd_global.rand(), rd_global.rand());
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
  float delta_q = 0.3f * h;
  float r = glm::length(p_i.pos - p_j.pos);
  return -k * fast_pow(poly6(r, h) / poly6(delta_q, h), n);
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
  int i = threadIdx.x + blockIdx.x * blockDim.x;
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
  int i = threadIdx.x + blockIdx.x * blockDim.x;

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
