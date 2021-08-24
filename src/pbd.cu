//
// Created by kr2 on 8/10/21.
//

#include "cu_common.cuh"
#include "pbd.hpp"
#include <chrono>
#include <iostream>

PBDSolver::PBDSolver(float _radius)
    : ch_ptr(std::make_shared<NSearch>(_radius)),
      _radius(_radius),
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

  hvector<SPHParticle> &data = get_data();
  const hvector<SPHParticle> pre_data = data;  // copy
  const int data_size = data.size();

  // Apply forces
  thrust::for_each(data.begin(), data.end(), [&](SPHParticle &item) {
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

  hvector<float> _lambda(data_size), c_i(data_size);
  int iter_cnt = iter;
  while (iter_cnt--) {
    for (int i = 0; i < data_size; ++i) {
      const auto &neighbor_vec = ch_ptr->neighbor_vec(i);

      // --- calculate rho ---
      float rho = 0.0f;
      for (const auto &j : neighbor_vec)
        rho += mass * poly6(glm::length(data[i].pos - data[j].pos), _radius);

      c_i[i] = glm::max(rho / rho_0 - 1, 0.0f);

      // --- calculate grad_c ---
      float _denom = 0.0f;
      for (const auto &j : neighbor_vec) {
        vec3 grad_c(0.0f);
        if (i == j) {
          for (int k = 0; k < ch_ptr->n_neighbor(i); ++k) {
            const int neighbor_index = ch_ptr->neighbor(i, k);
            grad_c += grad_spiky(data[i].pos - data[neighbor_index].pos,
                                 _radius);
          }
        } else {
          grad_c = -grad_spiky(data[i].pos - data[j].pos, _radius);
        }

        grad_c = 1.0f / rho_0 * grad_c;
        _denom += fast_pow(glm::length(grad_c), 2.0f);
      }

      _lambda[i] = -c_i[i] / (_denom + denom_epsilon);
    }

    for (int i = 0; i < data_size; ++i) {
      vec3 delta_p_i(0.0f);
      const auto &neighbor_vec = ch_ptr->neighbor_vec(i);
      for (const auto &j : neighbor_vec) {
        delta_p_i += (_lambda[i] + _lambda[j] +
                      compute_s_corr(data[i], data[j], radius)) *
                     grad_spiky(data[i].pos - data[j].pos, _radius);
      }

      delta_p_i *= 1.0f / rho_0;
      data[i].pos += delta_p_i;
      data[i].rho = glm::clamp(c_i[i], 0.0f, 1.0f);

      constraint_to_border(data[i]);
    }
  }

  // update all velocity
  for (int i = 0; i < data_size; ++i) {
    auto &p = data[i];
    p.v = 1.0f / delta_t * (p.pos - pre_data[i].pos);
  }

  gui_ptr->set_particles(get_data());

  // Logging part
  if ((--interval) != 0)
    return;

  float c_i_sum = thrust::reduce(
      c_i.begin(), c_i.end(), 0.0f, thrust::plus<float>());

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
