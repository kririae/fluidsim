//
// Created by kr2 on 8/9/21.
//

#ifndef PBF3D__PARTICLE_HPP_
#define PBF3D__PARTICLE_HPP_

#include "common.hpp"
#include <memory>
#include <vector>

class Particle {
 public:
  vec3 pos;
  explicit Particle(vec3 _pos);
  Particle(float x, float y, float z);
  virtual ~Particle() = default;
  [[nodiscard]] float dist2(const Particle &p) const noexcept;
  [[nodiscard]] float dist(const Particle &p) const noexcept;

 protected:
};

class SPHParticle : public Particle {
 public:
  float rho{0};
  vec3 v{0};
  SPHParticle(vec3 _pos);
  SPHParticle(float x, float y, float z);
  SPHParticle(const SPHParticle &p) = default;
  SPHParticle &operator=(const SPHParticle &p) = default;
  ~SPHParticle() override = default;

 private:
};

#endif  // PBF3D__PARTICLE_HPP_
