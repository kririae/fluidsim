//
// Created by kr2 on 8/10/21.
//

#ifndef PBF3D_SRC_PBD_HPP_
#define PBF3D_SRC_PBD_HPP_

#include "gui.hpp"
#include "solver.hpp"
#include <memory>
#include <vector>

class SPHParticle;

class PBDSolver : public Solver {
 public:
  static constexpr int MAX_NEIGHBOR_SIZE = 0;
  float rho_0 = 10.0f;
  int iter = 3;
  vec3 ext_f = vec3(0.0f, -9.8f, 0.0f);

  explicit PBDSolver(float _radius);
  PBDSolver(const PBDSolver &solver) = delete;
  PBDSolver &operator=(const PBDSolver &solver) = delete;
  ~PBDSolver() override = default;

  void set_gui(RTGUI_particles *gui) noexcept;
  void callback() override;  // gui_ptr required
  void add_particle(const SPHParticle &p);
  hvector<SPHParticle> &get_data();

  // PBF mathematics parts...
  CUDA_FUNC_DEC static float poly6(float r, float h) noexcept;
  CUDA_FUNC_DEC static vec3 grad_spiky(vec3 v, float h) noexcept;
  CUDA_FUNC_DEC static float compute_s_corr(const SPHParticle &p_i,
                                            const SPHParticle &p_j,
                                            float h);
  __attribute__((device)) static void constraint_to_border(SPHParticle &p,
                                                           float _border);

  // hash function
  [[nodiscard]] static CUDA_FUNC_DEC inline int hash(float x,
                                                     float y,
                                                     float z,
                                                     int n_grids);
  [[nodiscard]] static CUDA_FUNC_DEC inline int hash(const vec3 &p,

                                                     int n_grids);
  [[nodiscard]] static CUDA_FUNC_DEC inline int hash_from_grid(int u,
                                                               int v,
                                                               int w,
                                                               int n_grids);
  [[nodiscard]] static CUDA_FUNC_DEC inline int hash_from_grid(const ivec3 &p,
                                                               int n_grids);
  [[nodiscard]] static CUDA_FUNC_DEC inline ivec3 get_grid_index(const vec3 &p);

  RTGUI_particles *gui_ptr{nullptr};
  float radius, radius2, mass{0}, delta_t{1.0f / 60.0f};
  hvector<SPHParticle> data;
  float *_lambda{}, *c_i{};
  int *dev_neighbor_map{}, *dev_n_neighbor_map{},
      *hash_map{};  // data_size * MAX_NEIGHBOR_SIZE
  size_t pitch{0};
  int n_grids{};
};

static __attribute__((global)) void update_velocity(SPHParticle *dev_data,
                                                    SPHParticle *pre_data,
                                                    float delta_t,
                                                    int data_size,
                                                    float _border);

static __attribute__((global)) void apply_force(SPHParticle *dev_data,
                                                float delta_t,
                                                vec3 ext_f,
                                                int data_size,
                                                float _border);

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

// Allow access to hash function
static __attribute__((global)) void build_hash_map(SPHParticle *dev_data,
                                                   int data_size,
                                                   int *dev_hash_map,
                                                   int *dev_n_hash_map,
                                                   int n_grids,
                                                   size_t pitch_hash);

static __attribute__((global)) void build_neighbor_map(
    SPHParticle *dev_data,
    int data_size,
    int *dev_neighbor_map,
    int *dev_n_neighbor_map,
    const int *dev_hash_map,
    const int *dev_n_hash_map,
    int n_grids,
    size_t pitch,
    size_t pitch_hash);
#endif  // PBF3D_SRC_PBD_HPP_
