//
// Created by kr2 on 8/28/21.
//

#ifndef PBF3D_SRC_MESHER_HPP_
#define PBF3D_SRC_MESHER_HPP_

#include <openvdb/Types.h>

class SPHParticle;

// copied from OpenVDB, structure definition
class ParticleList {
 public:
  explicit ParticleList(const std::vector<SPHParticle> &_data);
  virtual ~ParticleList() = default;

  using PosType = openvdb::Vec3R;
  // Return the total number of particles in the list.
  // Always required!
  size_t size() const;

  // Get the world-space position of the nth particle.
  // Required by rasterizeSpheres().
  void getPos(size_t n, openvdb::Vec3R &xyz) const;

  // Get the world-space position and radius of the nth particle.
  // Required by rasterizeSpheres().
  void getPosRad(size_t n, openvdb::Vec3R &xyz, openvdb::Real &radius) const;

  // Get the world-space position, radius and velocity of the nth particle.
  // Required by rasterizeTrails().

  // void getPosRadVel(size_t n,
  //                   openvdb::Vec3R &xyz,
  //                   openvdb::Real &radius,
  //                   openvdb::Vec3R &velocity) const;

  // Get the value of the nth particle's user-defined attribute (of type @c
  // AttributeType). Required only if attribute transfer is enabled in
  // ParticlesToLevelSet.
  void getAtt(size_t n, openvdb::Index32 &att) const;

 private:
  const std::vector<SPHParticle> &data;
};

// Use `data` as input, write result into points and triangles
void particleToMesh(const std::vector<SPHParticle> &data,
                    std::vector<openvdb::Vec3s> &points,
                    std::vector<openvdb::Vec3I> &triangles,
                    std::vector<openvdb::Vec4I> &quads);

#endif  // PBF3D_SRC_MESHER_HPP_
