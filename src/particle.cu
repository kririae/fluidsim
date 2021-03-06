//
// Created by kr2 on 8/9/21.
//

#include "common.hpp"
#include "particle.hpp"

CUDA_FUNC_DEC P_CONSTEXPR Particle::Particle(vec3 _pos) : pos(_pos)
{
}

CUDA_FUNC_DEC P_CONSTEXPR Particle::Particle(float x, float y, float z)
    : pos(x, y, z)
{
}

CUDA_FUNC_DEC P_CONSTEXPR float Particle::dist2(
    const Particle &p) const noexcept
{
  const float delta_x = p.pos.x - pos.x;
  const float delta_y = p.pos.y - pos.y;
  const float delta_z = p.pos.z - pos.z;
  return delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
}

CUDA_FUNC_DEC P_CONSTEXPR float Particle::dist(const Particle &p) const noexcept
{
  return glm::sqrt(Particle::dist2(p));
}

CUDA_FUNC_DEC P_CONSTEXPR SPHParticle::SPHParticle(vec3 _pos)
    : Particle::Particle(_pos)
{
}

CUDA_FUNC_DEC P_CONSTEXPR SPHParticle::SPHParticle(float x, float y, float z)
    : Particle(x, y, z)
{
}

CUDA_FUNC_DEC P_CONSTEXPR SPHParticle::SPHParticle(const SPHParticle &p)
    : Particle::Particle(p.pos)
{
  rho = p.rho;
  v = p.v;
}

CUDA_FUNC_DEC P_CONSTEXPR SPHParticle &SPHParticle::operator=(
    const SPHParticle &p)
{
  pos = p.pos;
  rho = p.rho;
  v = p.v;
  return *this;
}