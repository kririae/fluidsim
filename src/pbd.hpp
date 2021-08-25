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
  hvector<SPHParticle> &get_data();
  static void constraint_to_border(SPHParticle &p);

  // PBF mathematics parts...
  CUDA_FUNC_DEC static float poly6(float r, float h) noexcept;
  CUDA_FUNC_DEC static vec3 grad_spiky(vec3 v, float h) noexcept;
  CUDA_FUNC_DEC static float compute_s_corr(const SPHParticle &p_i,
                                            const SPHParticle &p_j,
                                            float h);

 private:
  RTGUI_particles *gui_ptr{nullptr};
  float radius, radius2, mass{0}, delta_t{1.0f / 60.0f};
};

static __attribute__((global)) void fill_lambda(SPHParticle *dev_data,
                                                int *dev_neighbor_map,
                                                size_t pitch,
                                                const int *dev_n_neighbor_map,
                                                float *_lambda,
                                                float *c_i,
                                                size_t data_size,
                                                float rho_0,
                                                float _radius,
                                                float mass);
static __attribute__((global)) void apply_motion(SPHParticle *dev_data,
                                                 const int *dev_neighbor_map,
                                                 size_t pitch,
                                                 const int *dev_n_neighbor_map,
                                                 const float *_lambda,
                                                 const float *c_i,
                                                 size_t data_size,
                                                 float rho_0,
                                                 float _radius);

#endif  // PBF3D_SRC_PBD_HPP_
