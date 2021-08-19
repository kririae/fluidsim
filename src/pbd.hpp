//
// Created by kr2 on 8/10/21.
//

#ifndef PBF3D_SRC_PBD_HPP_
#define PBF3D_SRC_PBD_HPP_

#include "gui.hpp"
#include "nsearch.hpp"
#include "solver.hpp"
#include <memory>
#include <vector>

class SPHParticle;

class PBDSolver : public Solver {
 public:
  float rho_0 = 10.0f;
  float denom_epsilon = 20.0f;
  int iter = 3;
  vec3 ext_f = vec3(0.0f, -9.8f, 0.0f);
  std::shared_ptr<NSearch> ch_ptr;  // TODO: expose the interface temporary

  explicit PBDSolver(float _radius);
  PBDSolver(const PBDSolver &solver) = delete;
  PBDSolver &operator=(const PBDSolver &solver) = delete;
  ~PBDSolver() override = default;

  void set_gui(RTGUI_particles *gui) noexcept;
  void callback() override;  // gui_ptr required
  void add_particle(const SPHParticle &p);
  static void constraint_to_border(SPHParticle &p);
  vector<SPHParticle> &get_data();

  // PBF mathematics parts...
  float sph_calc_rho(int p_i);
  vec3 grad_c(int p_i, int p_k);
  static float poly6(float r, float d) noexcept;
  static vec3 grad_spiky(vec3 v, float d) noexcept;
  inline float compute_s_corr(int p_i, int p_j);

 private:
  RTGUI_particles *gui_ptr{nullptr};
  float radius, radius2, mass{0}, delta_t{1.0f / 60.0f};
};

#endif  // PBF3D_SRC_PBD_HPP_
