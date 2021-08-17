//
// Created by kr2 on 8/11/21.
//

#ifndef PBF3D_SRC_SOLVER_HPP_
#define PBF3D_SRC_SOLVER_HPP_

class Solver {
 public:
  Solver() = default;
  Solver(const Solver &solver) = delete;
  Solver &operator=(const Solver &solver) = delete;
  virtual ~Solver() = default;

  virtual void callback();
};

#endif  // PBF3D_SRC_SOLVER_HPP_
