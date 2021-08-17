//
// Created by kr2 on 8/10/21.
//

#ifndef PBF3D_SRC_PBD_CUH_
#define PBF3D_SRC_PBD_CUH_

#include "gui.cuh"
#include "nsearch.cuh"
#include "solver.hpp"
#include <memory>
#include <thrust/host_vector.h>
#include <vector>

class SPHParticle;

class PBDSolver : public Solver {
 public:
  std::shared_ptr<NSearch> ch_ptr;

  explicit PBDSolver(float _radius);
  PBDSolver(const PBDSolver &solver) = delete;
  PBDSolver &operator=(const PBDSolver &solver) = delete;
  ~PBDSolver() override = default;

  void set_gui(RTGUI_particles *gui) noexcept;
  void callback() override;  // gui_ptr required
  void add_particle(const SPHParticle &p);
  static void constraint_to_border(SPHParticle &p);
  void sync_data_from_nsearch();
  void sync_data_to_nsearch();
  [[nodiscard]] const dvector<SPHParticle> &get_data() const;

  // PBF mathematics parts...
  __device__ float sph_calc_rho(int p_i);
  __device__ vec3 grad_c(int p_i, int p_k);
  __device__ static float poly6(float r, float d) noexcept;
  __device__ static vec3 grad_spiky(vec3 v, float d) noexcept;
  __device__ float compute_s_corr(int p_i, int p_j) const;

 private:
  __device__ dvector<SPHParticle> data{};
  RTGUI_particles *gui_ptr{nullptr};
  float radius, radius2, mass{0}, delta_t{1.0f / 60.0f};
};

#endif  // PBF3D_SRC_PBD_CUH_
