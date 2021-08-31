//
// Created by kr2 on 8/11/21.
//

#ifndef PBF3D_SRC_SOLVER_HPP_
#define PBF3D_SRC_SOLVER_HPP_

class RTGUI_particles;
class SPHParticle;

class Solver {
 public:
  static constexpr int n_substeps = 1;

  Solver() = default;
  Solver(const Solver &solver) = delete;
  Solver &operator=(const Solver &solver) = delete;
  virtual ~Solver() = default;

  virtual void update_gui(RTGUI_particles *gui_ptr) noexcept;
  virtual void substep();
  virtual void add_particle(const SPHParticle &p);
};

#endif  // PBF3D_SRC_SOLVER_HPP_
