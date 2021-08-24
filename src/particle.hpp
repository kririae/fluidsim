//
// Created by kr2 on 8/9/21.
//

#ifndef PBF3D__PARTICLE_HPP_
#define PBF3D__PARTICLE_HPP_

#include "common.hpp"
#include <memory>
#include <vector>

// #define CUDA_FUNC_DECL
#define P_CONSTEXPR

class Particle {
 public:
  vec3 pos{};
  CUDA_FUNC_DEC P_CONSTEXPR Particle() = default;
  CUDA_FUNC_DEC P_CONSTEXPR explicit Particle(vec3 _pos);
  CUDA_FUNC_DEC P_CONSTEXPR Particle(float x, float y, float z);
  CUDA_FUNC_DEC ~Particle() =
      default;  // should not be virtual(in case of CUDA)
  [[nodiscard]] CUDA_FUNC_DEC P_CONSTEXPR float dist2(
      const Particle &p) const noexcept;
  [[nodiscard]] CUDA_FUNC_DEC P_CONSTEXPR float dist(
      const Particle &p) const noexcept;

 protected:
};

class SPHParticle : public Particle {
 public:
  float rho{0};
  vec3 v{0};
  CUDA_FUNC_DEC P_CONSTEXPR SPHParticle() = default;
  CUDA_FUNC_DEC P_CONSTEXPR explicit SPHParticle(vec3 _pos);
  CUDA_FUNC_DEC P_CONSTEXPR SPHParticle(float x, float y, float z);
  CUDA_FUNC_DEC P_CONSTEXPR SPHParticle(const SPHParticle &p);
  CUDA_FUNC_DEC P_CONSTEXPR SPHParticle &operator=(const SPHParticle &p);
  CUDA_FUNC_DEC ~SPHParticle() = default;

 private:
};

#endif  // PBF3D__PARTICLE_HPP_
