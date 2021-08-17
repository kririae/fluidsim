//
// Created by kr2 on 8/9/21.
//

#include "particle.hpp"
#include "common.hpp"

Particle::Particle(vec3 _pos) : pos(_pos)
{
}

Particle::Particle(float x, float y, float z) : pos(x, y, z)
{
}

float Particle::dist2(const Particle &p) const noexcept
{
  const float delta_x = p.pos.x - pos.x;
  const float delta_y = p.pos.y - pos.y;
  const float delta_z = p.pos.z - pos.z;
  return delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
}

float Particle::dist(const Particle &p) const noexcept
{
  return glm::sqrt(Particle::dist2(p));
  // return glm::length(pos - p.pos);
}

SPHParticle::SPHParticle(vec3 _pos) : Particle::Particle(_pos)
{
}

SPHParticle::SPHParticle(float x, float y, float z) : Particle(x, y, z)
{
}
