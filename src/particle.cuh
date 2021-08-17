//
// Created by kr2 on 8/9/21.
//

#ifndef PBF3D__PARTICLE_HPP_
#define PBF3D__PARTICLE_HPP_

#include "common.cuh"
#include <memory>
#include <vector>

#define P_FUNC_DECL __attribute__((host)) __attribute__((device))
#define P_CONSTEXPR

class Particle {
 public:
  vec3 pos;
  P_FUNC_DECL P_CONSTEXPR explicit Particle(vec3 _pos);
  P_FUNC_DECL P_CONSTEXPR Particle(float x, float y, float z);
  P_FUNC_DECL virtual ~Particle() = default;
  [[nodiscard]] P_FUNC_DECL P_CONSTEXPR float dist2(
      const Particle &p) const noexcept;
  [[nodiscard]] P_FUNC_DECL P_CONSTEXPR float dist(
      const Particle &p) const noexcept;

 protected:
};

class SPHParticle : public Particle {
 public:
  float rho{0};
  vec3 v{0};
  P_FUNC_DECL P_CONSTEXPR explicit SPHParticle(vec3 _pos);
  P_FUNC_DECL P_CONSTEXPR SPHParticle(float x, float y, float z);
  P_FUNC_DECL P_CONSTEXPR SPHParticle(const SPHParticle &p);
  P_FUNC_DECL P_CONSTEXPR SPHParticle &operator=(const SPHParticle &p);
  P_FUNC_DECL ~SPHParticle() override = default;

 private:
};

#endif  // PBF3D__PARTICLE_HPP_
